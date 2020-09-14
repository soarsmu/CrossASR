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

from simpletransformers.classification import ClassificationModel

import torch

cuda_available = torch.cuda.is_available()
print("Cuda available: ", cuda_available)

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


def create_2_label(df):
    df_2_label = df.copy()
    succes_test_case = np.where(df_2_label["label"] == constant.SUCCESS_TEST_CASE)[0]
    df_2_label["label"][succes_test_case] = constant.DETERMINED_TEST_CASE
    
    return df_2_label

def different(raw_outputs) :
    diff = []
    for pred in raw_outputs :
        diff.append(pred[1] - pred[0])
    return np.array(diff)

def intersection(lst1, lst2): 
    lst3 = [value for value in lst1 if value in lst2] 
    return lst3

def substraction(l1, l2) :
    l3 = [x for x in l1 if x not in l2]
    return l3

def merge_asr_data(data):
    _sr = "wit"
    stc = {}
    for sr in ASR :
        stc[sr] = np.where(data[sr]["label"] == constant.SUCCESS_TEST_CASE)[0]

    intersection_stc = stc[_sr]
    for sr in ASR :
        intersection_stc = intersection(intersection_stc, stc[sr])

    utc = np.where(data[_sr]["label"] == constant.UNDETERMINED_TEST_CASE)[0]
    idx = data[_sr].index.values.tolist()
    ftc = substraction(idx, intersection_stc)
    ftc = substraction(idx, utc)

    merged = pd.DataFrame(columns=["sentence", "label"])

    for id in ftc :
        merged = merged.append({"sentence" : data[_sr]["sentence"][id], "label" : 1}, ignore_index=True)

    for id in intersection_stc :
        merged = merged.append({"sentence" : data[_sr]["sentence"][id], "label" : 0}, ignore_index=True)

    merged = merged.sample(frac=1, random_state=constant.INITIAL_SEED).reset_index(drop=True)
    
    return merged



def printHelp() :
    print('with_classifier.py -s <random seed> -n <number of batch> -b <batch size> -t <batch-time>')
    print('or')
    print('with_classifier.py --seed <random seed> --number-of-batch <number of batch> --batch-size <batch size> --batch-time <batch time>')

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

    APPROACH = "with_classifier"

    fix_corpus = getCorpus(constant.CORPUS_FPATH)
    
    print("Seed: ", random_seed)
    print("Approach: ", APPROACH)
    print("Corpus: ", constant.CORPUS_FPATH)
    print("Number of Batch: ", n_batch)
    print("Batch Size: ", batch_size)
    print("Batch Time: ", batch_time)

    for tts in TTS :
        
        model_1_args = {
                "output_dir": "models/albert/" + tts + "/two_stage/1/",
                "overwrite_output_dir" : True
            }
        
        # Create a ClassificationModel for First Stage Classifier
        model_1 = ClassificationModel("albert", "albert-base-v2", args=model_1_args, use_cuda=cuda_available)

        model_2_args = {
            "output_dir": "models/albert/" + tts + "/two_stage/2/",
            "overwrite_output_dir" : True
        }
        # Create a ClassificationModel for Second Stage Classifier
        model_2 = ClassificationModel("albert", "albert-base-v2", args=model_2_args, use_cuda=cuda_available)


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

        audio_dir = "audio/%s/%s-%d/%s/" % (APPROACH, constant.DATASET, random_seed, tts)
        if not os.path.exists(audio_dir):
            os.makedirs(audio_dir)

        i_batch = 0
        while i_batch < n_batch :
            
            i_batch += 1 
            
            curr_data = {}
            for sr in ASR :
                curr_data[sr] = pd.DataFrame(columns=["sentence", "label"])
            
            lower_bound = 0
            upper_bound = batch_size
            
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
                    
                

                df = pd.DataFrame(corpus[upper_bound:], columns=["id", "sentence"])

                start_time = time.time()

                # stage 1
                df2 = create_2_label(data["wit"])
                model_1.train_model(df2, show_running_loss=False, verbose=False)
                predictions, raw_outputs = model_1.predict(df["sentence"].values)

                df["probability"] = different(raw_outputs)
                df = df.sort_values(by=['probability'], ascending=False).reset_index(drop=True)

                top_threshold = 400
                processed = df[:top_threshold]
                unprocessed = df[top_threshold:] 

                training_time = time.time() - start_time
                print("Training time 1: %.2f" % training_time)
    
                # stage 2
                merged_data = merge_asr_data(data)
                model_2.train_model(merged_data, show_running_loss=False, verbose=False)
                predictions_2, raw_outputs_2 = model_2.predict(processed["sentence"].values)

                processed["probability"] = different(raw_outputs_2)
                processed = processed.sort_values(by=['probability'], ascending=False).reset_index(drop=True)

                processed = processed.drop(columns=["probability"])

                df = processed.append(unprocessed)

                training_time = time.time() - start_time
                print("Training time all: %.2f" % training_time)
                
                corpus = []
                for index, row in df.iterrows():
                    corpus.append({"id": row['id'], "sentence": row['sentence']})
                    
                    
        for sr in ASR :
            data[sr] = data[sr].reset_index(drop=True)
        
        # save the result
        for sr in ASR :
            fpath = "result/%s/%s-%d/%s/%s/" % (APPROACH, constant.DATASET, random_seed, tts, sr)
            if not os.path.exists(fpath):
                os.makedirs(fpath)
            stat[sr].to_csv(fpath + "statistic.csv", index=False)
            data[sr].to_csv(fpath + "data.csv", index=False)
#             print(stat[sr])

if __name__ == "__main__":
    main(sys.argv[1:])
