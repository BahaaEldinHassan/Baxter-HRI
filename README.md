# Robot Human Interaction

## Computer Vision

To run the vision modules, we need `Python 3.11+` (with `pip`). The other required Python packages are described in `requirements.txt`. Before executing any of the modules described in this section, make sure to be using this Python version and have the requirements installed.

Note: You can install the required Python packages with this command (run this from the root of the project directory):

```shell
pip install -r requirements.txt
```

Then change the directory to `ml_nn_vision`:

```shell
cd ml_nn_vision
```

We have two modules for the vision part. Their functionalities and how to run them are described below:

### Savant

This module includes the CNN classification model used to train the model. It also preprocesses the dataset and takes care of data padding (e.g., the hand and body gestures have different dimensionality, to match it to each other this module pads the lower dimensional one with some constant values).

To run the module, use the following command:

```shell
python -m savant -e <epoch_count> -lr <learning_rate> -cp <checkpoint_filename>
```

All the available CLI options are as follows:

```shell
Usage: python -m savant [OPTIONS]

Options:
  -e, --num-epochs INTEGER    Number of epochs.
  -lr, --learning-rate FLOAT  Learning rate.
  -d, --dataset [combined]    Dataset to use.
  -M, --model [cnn]           Neural network model to use.
  -m, --momentum FLOAT        SGD Momentum.
  -g, --num-gates INTEGER     Number of gates to use (for GRU).
  -cp, --checkpoint TEXT      Checkpoint file to use.
  --help                      Show this message and exit.
```

### Wisdom

This module is used to take care of the recording and collection of test data, use trained model with multiple camera feeds (Intel RealSense, laptop camera module etc) to infer in real time, depth testing with YOLOv8 object detection.

#### Recording and Training Data Collection

To collect the data, you can use this command:

```shell
python -m wisdom record -l thumbs_up -cf camera -r hand
```

All the available commands are as follows:

```shell
Usage: python -m wisdom record [OPTIONS]

Options:
  -l, --label [thinking|victory|beckoning|closed_fist|handshake|open_palm|thumbs_down|thumbs_up]
                                  Label to use.  [required]
  -cf, --camera-feed [camera|realsense]
                                  Camera feed to use.  [required]
  -r, --recorder [body|hand|yolo]
                                  Recorder to use.  [required]
  --help                          Show this message and exit.
```

#### RealTime Inference

To run real-time inference, you can use this command:

```shell
python -m wisdom infer -cp <checkpoint_filename>
```

Note: Make sure you have trained the model and already have a model checkpoint file.

All the available commands are as follows:

```shell
Usage: python -m wisdom infer [OPTIONS]

Options:
  -cf, --camera-feed [camera|realsense]
                                  Camera feed to use.  [required]
  -cp, --checkpoint TEXT          Checkpoint file to use.  [required]
  --help                          Show this message and exit.
```

#### RealTime Depth Inference with YOLOv8

To run the real-time depth inference using the Intel RealSense camera with YOLO, you can use this command:

```shell
python -m wisdom record -l thumbs_up -cf realsense -r yolo
```

