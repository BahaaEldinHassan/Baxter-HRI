import torch
import torchaudio

from torchaudio.datasets import SPEECHCOMMANDS
import os
from tqdm import tqdm


class SubsetSC(SPEECHCOMMANDS):
    def __init__(self, subset: str = None):
        super().__init__("./", download=True)

        if subset == "validation":
            self._walker = self.load_list("validation_list.txt")
        elif subset == "testing":
            self._walker = self.load_list("testing_list.txt")
        elif subset == "training":
            excludes = self.load_list("validation_list.txt") + self.load_list("testing_list.txt")
            excludes = set(excludes)
            self._walker = [w for w in self._walker if w not in excludes]

    def load_list(self, filename):
            filepath = os.path.join(self._path, filename)
            with open(filepath) as fileobj:
                return [os.path.normpath(os.path.join(self._path, line.strip())) for line in fileobj]


class CNN(torch.nn.Module):
    def __init__(self, n_input=1, n_output=35, stride=16, n_channel=32):
        super().__init__()
        self.conv1 = torch.nn.Conv1d(n_input, n_channel, kernel_size=80, stride=stride)
        self.bn1 = torch.nn.BatchNorm1d(n_channel)
        self.pool1 = torch.nn.MaxPool1d(4)
        self.conv2 = torch.nn.Conv1d(n_channel, n_channel, kernel_size=3)
        self.bn2 = torch.nn.BatchNorm1d(n_channel)
        self.pool2 = torch.nn.MaxPool1d(4)
        self.conv3 = torch.nn.Conv1d(n_channel, 2 * n_channel, kernel_size=3)
        self.bn3 = torch.nn.BatchNorm1d(2 * n_channel)
        self.pool3 = torch.nn.MaxPool1d(4)
        self.conv4 = torch.nn.Conv1d(2 * n_channel, 2 * n_channel, kernel_size=3)
        self.bn4 = torch.nn.BatchNorm1d(2 * n_channel)
        self.pool4 = torch.nn.MaxPool1d(4)
        self.fc1 = torch.nn.Linear(2 * n_channel, n_output)
        

    def forward(self, x):
        x = self.conv1(x)
        x = torch.nn.functional.relu(self.bn1(x))
        x = self.pool1(x)
        x = self.conv2(x)
        x = torch.nn.functional.relu(self.bn2(x))
        x = self.pool2(x)
        x = self.conv3(x)
        x = torch.nn.functional.relu(self.bn3(x))
        x = self.pool3(x)
        x = self.conv4(x)
        x = torch.nn.functional.relu(self.bn4(x))
        x = self.pool4(x)
        x = torch.nn.functional.avg_pool1d(x, x.shape[-1])
        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        return torch.nn.functional.log_softmax(x, dim=2)
    

class LSTM(torch.nn.Module):
    def __init__(self, n_input=1, n_output=35, stride=16, n_channel=32):
        super().__init__()
        self.lstm = torch.nn.LSTM(input_size=8000, hidden_size = n_output, num_layers=4)
        self.fc1 = torch.nn.Linear(n_output, n_output)
        

    def forward(self, x):
        x, _ = self.lstm(x)
        #x = torch.nn.functional.relu(x)
        x = self.fc1(x)
        return torch.nn.functional.log_softmax(x, dim=2)

class AudioModel:

    def __init__(self, batch_size = 256, modelType = "CNN") -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(self.device)
        self.train_set = SubsetSC("training")
        self.test_set = SubsetSC("testing")
        self.labels = sorted(list(set(datapoint[2] for datapoint in self.train_set)))
        print(self.labels)
        self.nbOfLabels = len(self.labels)

        if modelType=="CNN":
            self.modelType=CNN
        elif modelType=="LSTM":
            self.modelType = LSTM
        else:
            Exception("Model type not supported")

        if self.device == "cuda":
            num_workers = 1
            pin_memory = True
        else:
            num_workers = 0
            pin_memory = False

        self.train_loader = torch.utils.data.DataLoader(
            self.train_set,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=self.collate_fn,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        self.test_loader = torch.utils.data.DataLoader(
            self.test_set,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            collate_fn=self.collate_fn,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        waveform, sample_rate, label, speaker_id, utterance_number = self.train_set[0]
        self.transform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=8000)
        transformed = self.transform(waveform)

        self.model = self.modelType(n_input=transformed.shape[0], n_output=35)
        self.model.to(self.device)
        print(self.model)
    
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.05)#, weight_decay=0.0001)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=20, gamma=0.1)


    def pad_sequence(self, batch):
        # Make all tensor in a batch the same length by padding with zeros
        batch = [item.t() for item in batch]
        batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0.)
        return batch.permute(0, 2, 1)

    def collate_fn(self, batch):

        # A data tuple has the form:
        # waveform, sample_rate, label, speaker_id, utterance_number

        tensors, targets = [], []

        # Gather in lists, and encode labels as indices
        for waveform, _, label, *_ in batch:
            tensors += [waveform]
            targets += [torch.tensor(self.labels.index(label))
]

        # Group the list of tensors into a batched tensor
        tensors = self.pad_sequence(tensors)
        targets = torch.stack(targets)

        return tensors, targets

    def __train(self, model, epoch, log_interval, device):
        model.train()
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data = data.to(device)
            target = target.to(device)
            # apply transform and model on whole batch directly on device
            data = self.transform(data)
            output = model(data)
            # negative log-likelihood for a tensor of size (batch x 1 x n_output)
            loss = torch.nn.functional.nll_loss(output.squeeze(), target)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            # print training stats
            if batch_idx % log_interval == 0:
                print(f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(self.train_loader.dataset)} ({100. * batch_idx / len(self.train_loader):.0f}%)]\tLoss: {loss.item():.6f}")

            # update progress bar
            self.pbar.update(self.pbar_update)
            # record loss
            self.losses.append(loss.item())

    def __test(self, model, epoch, device):
        model.eval()
        correct = 0
        for data, target in self.test_loader:

            data = data.to(device)
            target = target.to(device)

            # apply transform and model on whole batch directly on device
            data = self.transform(data)
            output = model(data)

            pred = output.argmax(dim=-1)
            correct += pred.squeeze().eq(target).sum().item()

            # update progress bar
            self.pbar.update(self.pbar_update)

            print(f"\nTest Epoch: {epoch}\tAccuracy: {correct}/{len(self.test_loader.dataset)} ({100. * correct / len(self.test_loader.dataset):.0f}%)\n")

    def train(self, n_epoch = 2):
        device = self.device
        log_interval = 20

        self.pbar_update = 1 / (len(self.train_loader) + len(self.test_loader))
        self.losses = []

        # The transform needs to live on the same device as the model and the data.
        self.transform = self.transform.to(device)
        with tqdm(total=n_epoch) as self.pbar:
            for epoch in range(1, n_epoch + 1):
                self.__train(self.model, epoch, log_interval, device)
                self.__test(self.model, epoch, device)
                #self.scheduler.step()

        torch.save(self.model.state_dict(), "model.pth")
        print("Saved PyTorch Model State to model.pth")

    def loadModel(self, name = "model.pth"):
        self.model = self.modelType().to(self.device)
        self.model.load_state_dict(torch.load(name))
        print(self.model)

    def predict(self, tensor, resample = True):
        # Use the model to predict the label of the waveform
        if resample==False:
            tensor = torch.tensor([tensor])
            print(tensor.size())
        tensor = tensor.to(self.device)
        if resample==True:
            tensor = self.transform(tensor)
            print(tensor.size())
        tensor = self.model(tensor.unsqueeze(0))
        tensor = tensor.argmax(dim=-1)
        tensor = self.labels[tensor.squeeze()]
        return tensor