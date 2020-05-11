import random
import queue
from datetime import datetime
import time
import os
import subprocess
import string
import math
import numpy as np

import utils, constant

import requests, urllib

import wave
from gtts import gTTS

from alexa_client import AlexaClient
from alexa_client.alexa_client import constants
from alexa_client.alexa_client import helpers

from wit import Wit

import speech_recognition

from gensim.models import Word2Vec


# Speech
import soundfile as sf  # pip install pysoundfile
# pip install python_speech_features
import python_speech_features as speech_lib

from jiwer import wer

import joblib

from datetime import datetime

class WaitTimeoutError(Exception):
    pass
class RequestError(Exception):
    pass
class UnknownValueError(Exception):
    pass


GOOGLE_TTS = "google"
APPLE_TTS = "apple"
TTS = GOOGLE_TTS

DEEPSPEECH = "deepspeech"
ALEXA = "alexa"
GCLOUD = "gcloud"
CHROMESPEECH = "gspeech"
WIT = "wit"
WAV2LETTER = "wav2letter"
PADDLEDEEPSPEECH = "paddledeepspeech"
SR = [DEEPSPEECH, WIT, WAV2LETTER, PADDLEDEEPSPEECH]

r = speech_recognition.Recognizer()

CLASSIFIER_MODEL = "RF"
classifier = {}
for sr in SR :
    classifier_fpath = "model/" + CLASSIFIER_MODEL + "_" + sr + ".sav"
    classifier[sr] = joblib.load(classifier_fpath)

TRANSFORMER_MODEL = "model/transformer.sav"
transformer = joblib.load(TRANSFORMER_MODEL)


# alexa 
client_id = os.getenv("CLIENT_ID")
client_secret = os.getenv("CLIENT_SECRET")
refresh_token = os.getenv("REFRESH_TOKEN")
BASE_URL_NORTH_AMERICA = 'alexa.na.gateway.devices.a2z.com'

# wit 
WIT_AI_KEY = "5PBOPP2VVZM3MJFQOKK57YRG4DFWXIBZ"

if ALEXA in SR :
    client = AlexaClient(
        client_id=client_id,
        secret=client_secret,
        refresh_token=refresh_token,
        base_url=BASE_URL_NORTH_AMERICA
    )

    client.connect()  # authenticate and other handshaking steps


def get_corpus(fpath) :
    corpus = []
    file = open(fpath)
    lines = file.readlines()
    id = 0
    for l in lines :
        id += 1
        corpus.append({"id": id, "text": l[:-1]})
    file.close()
    # random.shuffle(corpus)
    return corpus

def initiate_cases() :

    cases = {}

    for sr in SR :
        fpath = "training_data/" + sr + ".txt"
        file = open(fpath)
        lines = file.readlines()
        id = 0
        case = {}
        for l in lines:
            id += 1
            case[id] = int(l.split(",")[-1])
        file.close()

        cases[sr] = case

    return cases

def get_case(cases, id) :
    case = {}
    
    for sr in SR :
        case[sr] = cases[sr][id]

    return case

def classify_bert(text):
    text = utils.preprocess_text(text)
    resp = requests.get(
        'http://10.4.4.55:5000/translate?text=' + urllib.parse.quote(text))
    if resp.status_code != 200:
        raise 'GET /translate/ {}'.format(resp.status_code)
    return int(resp.content.decode("utf-8"))
    

def initiate_folders() :
    folders = []
    main_folder = "guided_data/"
    folders.append(main_folder)
    mp3_google_tts_folder =  main_folder + "mp3/"
    folders.append(mp3_google_tts_folder)
    alexa_data_folder = main_folder + "alexa_wav/"
    folders.append(alexa_data_folder)
    wav_folder = main_folder + "wav/"
    folders.append(wav_folder)

    for folder in folders :
        if not os.path.exists(folder):
            os.makedirs(folder)


if __name__ == '__main__' :  

    initiate_folders()

    fpath = "corpus-sentence.txt"
    corpus = get_corpus(fpath)

    test_corpus = []
    train_size = math.ceil(len(corpus) * 3 / 4)
    for i in range(train_size, len(corpus)):
    # for i in range(0, train_size):
        test_corpus.append(corpus[i])

    cases = initiate_cases()

    time_max = 60
    
    x = 0
    while x < 3 : 
        x += 1
        
        corpus = test_corpus.copy()

        # shuffle the data
        random.seed(100 + x)
        random.shuffle(corpus)

        # using queue to process data one by one
        q = queue.Queue()
        for data in corpus :
            q.put(data)

        detected = []
        
        bugs = {}
        
        start_time = time.time()
        current_bug = 0
        last_time = 0

        i = 0 # number of texts processed
        j = 0 # number of predicted
        while (not q.empty() and last_time <= time_max) :
            i += 1
            data = q.get()           

            is_predicted_bug = classify_bert(data["text"])
            if (is_predicted_bug) :
                j += 1
                case = get_case(cases, data["id"])
                if constant.UNDETERMINED_LABEL in case.values() :
                    print("Can't determine bug")
                elif constant.BUG_LABEL not in case.values():
                    print("All Speech Recognition can recognize the speech")
                else :
                    # print("\n\n\n")
                    # print("text")
                    # print(data["text"])
                    # print("cases")
                    # print(case)
                    
                    current_bug += 1
                    time_execution = time.time() - start_time + j * 29.4815
                    t = math.ceil(time_execution / 60.0)
                    bug = {}
                    bug["time_execution"] = t
                    # bug["number_of_data"] = i
                    bug["number_of_bug"] = list(case.values()).count(constant.BUG_LABEL)
                    # bug["id_corpus"] = data["id"]
                    for sr in SR :
                        # change the label to make easier in cumulative calculation
                        if case[sr] == -1 :
                            case[sr] = 0

                    bug["case"] = case
                    
                    bugs[current_bug] = bug

                    last_time = t
        

        result = {}

        last_time = 0
        last_total_bug = 0
        last_bugs = {}
        for sr in SR: 
            last_bugs[sr] = 0

        for k, v in bugs.items() :
            for i in range(last_time, v["time_execution"]) :
                result[i+1] = {"number_of_bug": last_total_bug, "bug_per_asr": last_bugs.copy()}
            last_total_bug += v["number_of_bug"]
            for sr in SR:
                # cumulative calculation
                last_bugs[sr] += v["case"][sr] 
            result[v["time_execution"]] = {"number_of_bug": last_total_bug, "bug_per_asr" : last_bugs.copy()}
            last_time = v["time_execution"]
            
                    
                    
        file = open("result/estimated/2_label/with_classifier_" + str(time_max) + "_" +
                    str(datetime.now()) + ".txt", "w+")
        for k in result.keys():
            if k <= time_max :
                file.write("%d, %d, %d, %d, %d, %d\n" % (
                    k, result[k]["number_of_bug"], result[k]["bug_per_asr"][DEEPSPEECH], result[k]["bug_per_asr"][WIT], result[k]["bug_per_asr"][WAV2LETTER], result[k]["bug_per_asr"][PADDLEDEEPSPEECH]))
        
        file.close()



        # using queue to process data one by one
        q = queue.Queue()
        for data in corpus:
            q.put(data)

        detected = []

        bugs = {}

        start_time = time.time()
        current_bug = 0
        last_time = 0

        i = 0  # number of texts processed
        j = 0  # number of predicted
        while (not q.empty() and last_time <= time_max):
            i += 1
            data = q.get()
            case = get_case(cases, data["id"])
            j += 1
            case = get_case(cases, data["id"])
            if constant.UNDETERMINED_LABEL in case.values():
                print("Can't determine bug")
            elif constant.BUG_LABEL not in case.values():
                print("All Speech Recognition can recognize the speech")
            else:
                # print("\n\n\n")
                # print("text")
                # print(data["text"])
                # print("cases")
                # print(case)
                
                current_bug += 1
                time_execution = time.time() - start_time + j * 29.4815
                t = math.ceil(time_execution / 60.0)
                bug = {}
                bug["time_execution"] = t
                bug["number_of_bug"] = list(case.values()).count(constant.BUG_LABEL)
                # bug["id_corpus"] = data["id"]
                for sr in SR:
                    # change the label to make easier in cumulative calculation
                    if case[sr] == -1:
                        case[sr] = 0

                bug["case"] = case

                bugs[current_bug] = bug

                last_time = t

        result = {}

        last_time = 0
        last_total_bug = 0
        last_bugs = {}
        for sr in SR:
            last_bugs[sr] = 0

        for k, v in bugs.items():
            for i in range(last_time, v["time_execution"]):
                result[i+1] = {"number_of_bug": last_total_bug,
                               "bug_per_asr": last_bugs.copy()}
            last_total_bug += v["number_of_bug"]
            for sr in SR:
                # cumulative calculation
                last_bugs[sr] += v["case"][sr]
            result[v["time_execution"]] = {
                "number_of_bug": last_total_bug, "bug_per_asr": last_bugs.copy()}
            last_time = v["time_execution"]


        file = open("result/estimated/2_label/without_classifier_" + str(time_max) + "_" +
                    str(datetime.now()) + ".txt", "w+")
        for k in result.keys():
            if k <= time_max:
                file.write("%d, %d, %d, %d, %d, %d\n" % (
                    k, result[k]["number_of_bug"], result[k]["bug_per_asr"][DEEPSPEECH], result[k]["bug_per_asr"][WIT], result[k]["bug_per_asr"][WAV2LETTER], result[k]["bug_per_asr"][PADDLEDEEPSPEECH]))

        file.close()
    

