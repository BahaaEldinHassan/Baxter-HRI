import sounddevice as sd
import numpy as np
import queue
import scipy.io

class Audio:

    def __init__(self, samplerate = 8000, channels=1, device = sd.default.device):
        self.samplerate = samplerate
        self.channels = channels
        self.device = device
        self.__buffor = queue.Queue(self.samplerate*4)
        self.__stream = sd.InputStream(samplerate=self.samplerate, blocksize=None, device = self.device, 
                                     channels = self.channels, callback=self.__audioInputCallback)

    def startStream(self):
        buffSize = self.getBufforSize()
        self.getAudioInput(buffSize)
        self.__stream.start()

    def stopStream(self):
        self.__stream.stop()

    def __audioInputCallback(self, indata: np.ndarray, frames: int, time, status) -> None:
        for item in indata.flatten():
            self.__buffor.put_nowait(item)

    def getBufforSize(self):
        return self.__buffor.qsize()

    def getAudioInput(self, nbSamples) -> list:
        out = []
        for i in range(nbSamples):
            out.append(self.__buffor.get())
        return out
    
    def playAudio(self, data, samplerate = 16000):
        sd.play(data, samplerate=samplerate)
        sd.wait()

    def readWaveFile(self, path):
        rate, data = scipy.io.wavfile.read(path, mmap=False)
        return data, rate