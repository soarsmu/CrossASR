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
from time import sleep

from wit import Wit

from urllib.parse import urlencode
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError

from datetime import datetime

if __name__ == '__main__':

    translation_wit_writer = open(
        "output/wit_translation" + str(datetime.now()) + ".txt", "w+")

    dirpath = "data/tts_google/generated_speech/"
    WIT_AI_KEY = "5PBOPP2VVZM3MJFQOKK57YRG4DFWXIBZ"
    client = Wit(WIT_AI_KEY)
    for i in range(23041, 28540):
        filename = "audio_" + str(i) + ".wav"
        fpath = os.path.join(dirpath, filename)
        print("Processing: " + fpath)

        with open(fpath, 'rb') as audio:
            try :
                translation = None
                translation = client.speech(audio, None, {'Content-Type': 'audio/wav'})

                if translation != None :
                    if "_text" in translation:
                        sentence = str(translation["_text"])
                        print("Translation: " + sentence)
                        translation_wit_writer.write("%s\n" % (dirpath + ", " + filename[6:-4] + ", " + sentence))
                    else : 
                        translation_wit_writer.write("%s\n" % (dirpath + ", " + filename[6:-4] + ", "))
                else :
                    translation_wit_writer.write("%s\n" % (dirpath + ", " + filename[6:-4] + ", "))
            except Exception as e: 
                print("Could not request results from Wit.ai service; {0}".format(e))
            
            if (i % 100 == 0) :
                sleep(1)

    translation_wit_writer.close()
