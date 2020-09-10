import random
import queue
import os
import sys, getopt
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
# TTS = [constant.GOOGLE]
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



def printHelp() :
    print('without_classifier.py -s <random seed> -n <number of batch> -b <batch size> -t <batch-time>')
    print('or')
    print('without_classifier.py --seed <random seed> --number-of-batch <number of batch> --batch-size <batch size> --batch-time <batch time>')

def main(argv):
    random_seed = None
    n_batch = 5
    batch_size = 210
    batch_time = 60
    
    try:
        opts, args = getopt.getopt(argv,"hs:n:b:t:",["seed=", "number-of-batch=", "batch-size=", "batch-time="])
    except getopt.GetoptError:
        printHelp()
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            printHelp()
            sys.exit()
        elif opt in ("-s", "--seed"):
            random_seed = int(arg)
        elif opt in ("-n", "--number-of-batch"):
            n_batch = int(arg)
        elif opt in ("-b", "--batch-size"):
            batch_size = int(arg)
        elif opt in ("-t", "--batch-time"):
            batch_time = int(arg)
        
    if not random_seed :
        print("Please specify the seed number")
        sys.exit()
    
#     print("Random seed: ", random_seed)
#     print("Number of batch:", n_batch)
#     print("Batch size: ", batch_size)
#     print("Batch time: ", batch_time)

    APPROACH = "without_classifier"

    CORPUS_FPATH = "corpus/europarl-20k.txt"
    fix_corpus = getCorpus(CORPUS_FPATH)

    DATASET = "europarl"

    for tts in TTS :

        corpus = fix_corpus.copy()

        # shuffle the data
        random.seed(random_seed)
        random.shuffle(corpus)

        data = {}
        for sr in ASR :
            data[sr] = pd.DataFrame(columns=["sentence", "label"])

        stat = {}
        for sr in ASR :
            stat[sr] = pd.DataFrame(columns=["ftc", "stc", "utc"])

        audio_dir = "audio/%s/%s-%d/%s/" % (APPROACH, DATASET, random_seed, tts)
        if not os.path.exists(audio_dir):
            os.makedirs(audio_dir)

        i_batch = 0
        while i_batch < n_batch :
            
            curr_data = {}
            for sr in ASR :
                curr_data[sr] = pd.DataFrame(columns=["sentence", "label"])
            
            lower_bound = i_batch * batch_size 
            upper_bound = (i_batch + 1) * batch_size
            
            i_batch += 1 
            
            if lower_bound < len(corpus) :
                
                if upper_bound > len(corpus)-1 :
                    upper_bound = len(corpus)-1
            
                q = queue.Queue()
                for instance in corpus[lower_bound:upper_bound]:
                    q.put(instance)
                    
                start_time = time.time()
                last_time = 0

                while (not q.empty() and last_time <= batch_time):
                    instance = q.get()
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
                    data[sr] = data[sr].reset_index(drop=True)    

        # save the result
        for sr in ASR :
            fpath = "result/%s/%s-%d/%s/%s/" % (APPROACH, DATASET, random_seed, tts, sr)
            if not os.path.exists(fpath):
                os.makedirs(fpath)
            stat[sr].to_csv(fpath + "statistic.csv", index=False)
            data[sr].to_csv(fpath + "data.csv", index=False)
#             print(stat[sr])

if __name__ == "__main__":
    main(sys.argv[1:])
