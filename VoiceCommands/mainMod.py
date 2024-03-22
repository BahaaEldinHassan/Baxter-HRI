from spechToText import speechToText
from gpt import gpt
import os

commands = ["0.Nothing", "1. Start", "2. Stop", "3. Pick", "4. Place", "5. Pose"]
commandsDic = {commands[0] : 0,
                commands[1] : 1, 
                commands[2] : 2,
                commands[3] : 3,
                commands[4] : 4,
                commands[5] : 5}
keyword = "baxter"

chatGpt = gpt("".join(commands))
speech = speechToText()


def commandToLabel(command):
     return commandsDic[command]

def lookForKeyword(text):
     if keyword in text.lower():
          return True
     else:
          return False

def mainLoop():
      while(True):
        audio = speech.record_audio()
        text = speech.recognize_speech(audio)
        print("Recorded: ", text)
        response = "0.Nothing"
        if lookForKeyword(text):
            #response = chatGpt.makeRequest(text)
            print(response)
            print(commandToLabel(response))


if __name__=="__main__":
      mainLoop()