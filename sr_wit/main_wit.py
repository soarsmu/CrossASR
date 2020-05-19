#!/usr/bin/env python3
import os
import glob

# obtain path to "english.wav" in the same folder as this script
from os import path
import uuid
import time
from datetime import datetime
import base64
import hmac
import hashlib
import ssl
import certifi
import json
from time import sleep

from wit import Wit

from urllib.parse import urlencode
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError

from datetime import datetime

if __name__ == '__main__':

    TTS = "festival"
    SR = "wit"

    combination = "data/" + TTS + "/" + SR + "/"

    execution = combination + "execution_time/"
    transcription = combination + "transcription/"

    if not os.path.exists(combination):
        os.makedirs(combination)
    if not os.path.exists(execution):
        os.makedirs(execution)
    if not os.path.exists(transcription):
        os.makedirs(transcription)

    timestamp = str(datetime.now())

    file = open(execution + timestamp + ".txt", "w+")
    translation_wit_writer = open(transcription + timestamp + ".txt", "w+")

    dirpath = "data/" + TTS + "/generated_speech/"
    WIT_AI_KEY = "5PBOPP2VVZM3MJFQOKK57YRG4DFWXIBZ"
    client = Wit(WIT_AI_KEY)
    
    for i in range(1, 20001):

        start_time = time.time()

        filename = "audio_" + str(i) + ".wav"
        fpath = os.path.join(dirpath, filename)
        print("Processing: " + fpath)

        with open(fpath, 'rb') as audio:
            try :
                translation = None
                translation = client.speech(audio, None, {'Content-Type': 'audio/wav'})
                # print(translation)

                if translation != None :
                    if "text" in translation:
                        sentence = str(translation["text"])
                        print("Translation: " + sentence)
                        translation_wit_writer.write("%s\n" % (dirpath + ", " + filename[6:-4] + ", " + sentence))
                    else : 
                        translation_wit_writer.write("%s\n" % (dirpath + ", " + filename[6:-4] + ", "))
                else :
                    translation_wit_writer.write("%s\n" % (dirpath + ", " + filename[6:-4] + ", "))
            except Exception as e: 
                print("Could not request results from Wit.ai service; {0}".format(e))
            
            end_time = time.time()
            time_execution = round(end_time - start_time, 2)
            file.write("%d, %.2f\n" % (i, time_execution))

            sleep(0.1)

            if (i % 100 == 0) :
                sleep(1)

    file.close()

    translation_wit_writer.close()
