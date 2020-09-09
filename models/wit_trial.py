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

import os

ACCESS_TOKEN = os.getenv("WIT_ACCESS_TOKEN")

if __name__ == '__main__':

    client = Wit(ACCESS_TOKEN)
    
    fpath = "audio/google/hello.wav"
    print("Processing: " + fpath)

    with open(fpath, 'rb') as audio:
        try :
            translation = None
#             print(audio)
            translation = client.speech(audio, None, {'Content-Type': 'audio/wav'})
            print(translation)

            if translation != None :
                if "text" in translation:
                    sentence = str(translation["text"])
                    print("Translation: " + sentence)
        except Exception as e: 
            print("Could not request results from Wit.ai service; {0}".format(e))
