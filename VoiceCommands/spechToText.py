import speech_recognition as sr
from gpt import gpt
import os
import sys

class speechToText:

    polish = "pl"
    english = "en-GB"

    # Disable
    def blockPrint(self):
        sys.stdout = open(os.devnull, 'w')

    # Restore
    def enablePrint(self):
        sys.stdout = sys.__stdout__

    def __init__(self) -> None:
        self.r = sr.Recognizer()

    def record_audio(self):
        self.blockPrint()
        with sr.Microphone() as source:
            os.system('clear')
            audio = self.r.listen(source, timeout = 5, phrase_time_limit = 5)
            self.enablePrint()
        return audio

    def recognize_speech(self, audio):
        text = ""
        try:
            text = self.r.recognize_google(audio, language=self.english)
        except sr.UnknownValueError:
            print("I couldn't understand that.")
        except sr.RequestError:
            print("There was an error processing your request.")
        return text


