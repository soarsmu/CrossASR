#!/usr/bin/env python3
import os
import glob

# obtain path to "english.wav" in the same folder as this script
from os import path
import uuid
import time
import base64
import hmac
import hashlib
import ssl
import certifi
import json

import speech_recognition as sr
from time import sleep

from urllib.parse import urlencode
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError

from datetime import datetime

class WaitTimeoutError(Exception):
    pass

class RequestError(Exception):
    pass

class UnknownValueError(Exception):
    pass


if __name__ == '__main__':

    translation_gspeech_writer = open(
        "output/gspeech_translation" + str(datetime.now()) + ".txt", "w+")

    r = sr.Recognizer()

    # data = "data/"

    # for (dirpath, _, filenames) in os.walk(data):
    #     if (len(filenames) > 0):
    #         if not os.path.exists(dirpath):
    #             os.makedirs(dirpath)
    #         for i in range(1, len(filenames)+1):
    #             filename = "audio_" + str(i) + ".wav"
    #             if (filename in filenames):
    #                 fpath = os.path.join(dirpath, filename)
    #                 print("Processing: " + fpath)

    #                 # use the audio file as the audio source
    #                 with sr.AudioFile(fpath) as source:
    #                     audio = r.record(source)  # read the entire audio file

    #                 # recognize speech using Google Speech Recognition
    #                 try:
    #                     # for testing purposes, we're just using the default API key
    #                     # to use another API key, use `r.recognize_google(audio, key="GOOGLE_SPEECH_RECOGNITION_API_KEY")`
    #                     # instead of `r.recognize_google(audio)`
    #                     translation = r.recognize_google(audio)
    #                     translation_gspeech_writer.write(
    #                         "%s\n" % (dirpath + ", " + filename[6:-4] + ", " + translation))
    #                     print(
    #                         "Google Speech Recognition thinks you said " + translation)
    #                 except sr.UnknownValueError:
    #                     print("Google Speech Recognition could not understand audio")
    #                     translation_gspeech_writer.write(
    #                         "%s\n" % (dirpath + ", " + filename[6:-4] + ", "))
    #                 except sr.RequestError as e:
    #                     print(
    #                         "Could not request results from Google Speech Recognition service; {0}".format(e))

    dirpath = "data/tts_google/generated_speech/"
    for i in range(1, 21057):
        filename = "audio_" + str(i) + ".wav"
        fpath = os.path.join(dirpath, filename)
        print("Processing: " + fpath)

        # use the audio file as the audio source
        with sr.AudioFile(fpath) as source:
            audio = r.record(source)  # read the entire audio file

        # recognize speech using Google Speech Recognition
        try:
            # for testing purposes, we're just using the default API key
            # to use another API key, use `r.recognize_google(audio, key="GOOGLE_SPEECH_RECOGNITION_API_KEY")`
            # instead of `r.recognize_google(audio)`
            translation = r.recognize_google(audio)
            sleep(0.1)
            translation_gspeech_writer.write(
                "%s\n" % (dirpath + ", " + filename[6:-4] + ", " + translation))
            print(
                "Google Speech Recognition thinks you said " + translation)
        except sr.UnknownValueError:
            print("Google Speech Recognition could not understand audio")
            translation_gspeech_writer.write(
                "%s\n" % (dirpath + ", " + filename[6:-4] + ", "))
        except sr.RequestError as e:
            print(
                "Could not request results from Google Speech Recognition service; {0}".format(e))

        if (i % 100 == 0):
            sleep(20)



    translation_gspeech_writer.close()
