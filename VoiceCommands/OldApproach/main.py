from audio import Audio
from audioModel import SubsetSC, AudioModel
import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

y = np.zeros(8000)
fig, ax = plt.subplots()

def update(i, jakiesGowno, mic):
    global y, fig, ax
    buffSize = mic.getBufforSize()
    #print(buffSize)
    y = np.roll(y, -buffSize)
    y[-buffSize:] = mic.getAudioInput(buffSize)

    # Draw x and y lists
    ax.clear()
    ax.plot(y)

    # Format plot
    plt.ylim((-0.5, 0.5))
    plt.grid()
    plt.title('Microphone output')
    plt.ylabel('Raw value')

def main():
    #ani = animation.FuncAnimation(fig=fig, func=update, fargs=(2, mic), interval=100)
    #plt.show()

    #data, rate = mic.readWaveFile("OK.wav")
    #mic.playAudio(data, 16000)

    
    model = AudioModel(modelType="CNN")

    #model.train(10)
    model.loadModel("modelCNN.pth")

    waveform, sample_rate, label, speaker_id, utterance_number = model.test_set[40]
    mic = Audio(samplerate=8000)
    #mic.playAudio(waveform.numpy()[0], sample_rate)

    print(f"Expected: {label}. Predicted: {model.predict(waveform)}.")

'''
    while(True):
        input("SPACE")
        print("Recording..")
        mic.startStream()
        while(mic.getBufforSize()<8000):
            pass
        mic.stopStream()
        print("Stop recording...")
        waveform = mic.getAudioInput(8000)
        print(f"Predicted: {model.predict(waveform, False)}.")

'''

if __name__=="__main__":
    main()