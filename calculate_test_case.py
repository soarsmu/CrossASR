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
    print('calculate_test_case.py -t <tts name> -l <lower bound> -u <upper bound>')
    print('or')
    print('calculate_test_case.py --tts <tts name> --lower-bound <lower bound> --upper-bound <upper bound>')

def main(argv):
    tts = ""
    lower_bound = 0
    upper_bound = 20000
    
    try:
        opts, args = getopt.getopt(argv,"ht:l:u:",["tts=", "lower-bound=", "upper-bound="])
    except getopt.GetoptError:
        printHelp()
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            printHelp()
            sys.exit()
        elif opt in ("-t", "--tts"):
            tts = arg
        elif opt in ("-l", "--lower-bound"):
            lower_bound = int(arg)
        elif opt in ("-u", "--upper-bound"):
            upper_bound = int(arg)
        
    if tts == "":
        print("Please specify the seed number")
        sys.exit()
    
    APPROACH = "full"

    corpus = getCorpus(constant.CORPUS_FPATH)

    print("TTS: ", TTS)
    print("Approach: ", APPROACH)
    print("Corpus: ", constant.CORPUS_FPATH)

    data = {}
    stat = {}
    for sr in ASR :
        data[sr] = pd.DataFrame(columns=["sentence", "label"])
        stat[sr] = pd.DataFrame(columns=["ftc", "stc", "utc"])
    
    if lower_bound < 0 :
        print("Lower bound is less than zero")
        sys.exit()
    
    if lower_bound > len(corpus) :
        print("Lower bound is greater than the size of corpus")
        sys.exit()
    
    if upper_bound > len(corpus) :
        print("Upper bound is greater than the size of corpus")
        sys.exit()
    
    for i in range(lower_bound, upper_bound) :
        id = i + 1
        instance = corpus[id]
        transcriptions = utils.getTranscriptions(id)
        case = getCase(instance["sentence"], transcriptions)
        for sr in ASR :
            data[sr] = data[sr].append(
                {"sentence": instance["sentence"],
                    "label": case[sr]},
                ignore_index=True)

    for sr in ASR :
        ftc = len(np.where(data[sr]["label"] == constant.FAIL_TEST_CASE)[0])
        stc = len(np.where(data[sr]["label"] == constant.SUCCESS_TEST_CASE)[0])
        utc = len(np.where(data[sr]["label"] == constant.UNDETERMINED_TEST_CASE)[0])
        stat[sr] = stat[sr].append(
                    {"ftc": ftc, "stc": stc, "utc": utc}, 
                    ignore_index=True)

    # save the result
    for sr in ASR :
        fpath = "result/%s/%s/%s/%s/" % (APPROACH, constant.DATASET, tts, sr)
        if not os.path.exists(fpath):
            os.makedirs(fpath)
        stat[sr].to_csv(fpath + "statistic.csv", index=False)
        data[sr].to_csv(fpath + "data.csv", index=False)
#             print(stat[sr])

if __name__ == "__main__":
    main(sys.argv[1:])
