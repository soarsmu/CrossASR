import random
import queue
import os
import subprocess
import string
import math
import numpy as np
import pandas as pd

from datetime import datetime
import time

import requests
import urllib

import constant
import utils

TTS = constant.TTS
TTS = [constant.GOOGLE]
ASR = constant.ASR
# ASR = [constant.DEEPSPEECH, constant.PADDLEDEEPSPEECH, constant.WAV2LETTER]

def getTimestamp():
    now = datetime.now()
    date_time = now.strftime("%d_%m_%Y_%H_%M_%S")
    return date_time

def getCorpus(fpath):
    corpus = []
    file = open(fpath)
    lines = file.readlines()
    id = 0
    for l in lines:
        id += 1
        corpus.append({"id": id, "sentence": l[:-1]})
    file.close()
    # random.shuffle(corpus)
    return corpus


def generateSpeech(tts, text, fpath) :
    utils.synthesizeSpeech(tts, text, fpath)
        
def recognizeSpeech(tts, fpath) :
    
    transcriptions = {}
    
    for sr in ASR :
        transcriptions[sr] = utils.recognizeSpeech(sr, fpath)

    return transcriptions
    
def getCase(text, transcriptions) :
    
#     print(text)
#     print(transcriptions)
    
    case = {}
    success_count = 0
    for sr in ASR :
        transcription = utils.preprocess_text(transcriptions[sr])
        if text == transcription :
            case[sr] = constant.SUCCESS_TEST_CASE
            success_count += 1
        else :
            case[sr] = constant.FAIL_TEST_CASE
    
    if success_count == 0 :
        for sr in ASR :
            case[sr] = constant.UNDETERMINED_TEST_CASE
    
    return case


if __name__ == "__main__" :
    
    APPROACH = "without_classifier"

    CORPUS_FPATH = "corpus/europarl-20k.txt"
    fix_corpus = getCorpus(CORPUS_FPATH)

    DATASET = "europarl"

    for tts in TTS :
        
        time_max = 60

        x = 0
        while x < 3:
            x += 1

            corpus = fix_corpus.copy()

            # shuffle the data
            random_seed = constant.INITIAL_SEED + x
            random.seed(random_seed)
            random.shuffle(corpus)

            q = queue.Queue()
            for data in corpus:
                q.put(data)


            data = {}
            for sr in ASR :
                data[sr] = pd.DataFrame(columns=["sentence", "label"])

            stat = {}
            for sr in ASR :
                stat[sr] = pd.DataFrame(columns=["ftc", "stc", "utc"])

            n_batch = 5
            i_batch = 0

            training_time = 0
            
            audio_dir = "audio/%s/%s-%d/%s/" % (APPROACH, DATASET, random_seed, tts)
            if not os.path.exists(audio_dir):
                os.makedirs(audio_dir)
                

            while i_batch < n_batch :
                i_batch += 1 

                start_time = time.time()
                last_time = 0
                

                curr_data = {}
                for sr in ASR :
                    curr_data[sr] = pd.DataFrame(columns=["sentence", "label"])

                while (not q.empty() and last_time <= time_max):
                    instance = q.get()
#                     print("instance: ", instance)
                    fpath = audio_dir + "audio-" + str(instance["id"]) + ".wav"
                    generateSpeech(tts, instance["sentence"], fpath)
                    transcriptions = recognizeSpeech(tts, fpath)
                    case = getCase(instance["sentence"], transcriptions)
                    for sr in ASR :
                        curr_data[sr] = curr_data[sr].append(
                            {"sentence": instance["sentence"],
                                "label": case[sr]},
                            ignore_index=True)
                    
                    time_execution = time.time() - start_time
                    last_time = math.ceil(time_execution / 60.0)


                for sr in ASR :
                    ftc = len(np.where(curr_data[sr]["label"] == constant.FAIL_TEST_CASE)[0])
                    stc = len(np.where(curr_data[sr]["label"] == constant.SUCCESS_TEST_CASE)[0])
                    utc = len(np.where(curr_data[sr]["label"] == constant.UNDETERMINED_TEST_CASE)[0])
                    stat[sr] = stat[sr].append(
                                {"ftc": ftc, "stc": stc, "utc": utc}, 
                                ignore_index=True)

                    data[sr] = data[sr].append(curr_data[sr])


                for sr in ASR :
                    data[sr] = data[sr].reset_index(drop=True)    

            for sr in ASR :
                data[sr] = data[sr].reset_index(drop=True)

            # save the result
            for sr in ASR :
                fpath = "result/%s/%s-%d/%s/%s/" % (APPROACH, DATASET, random_seed, tts, sr)
                if not os.path.exists(fpath):
                    os.makedirs(fpath)
                stat[sr].to_csv(fpath + "statistic.csv", index=False)
                data[sr].to_csv(fpath + "data.csv", index=False)
                print(stat[sr])

