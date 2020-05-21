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

from datetime import datetime

TTS = constant.RV

SR = constant.SR

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
        fpath = "data/" + TTS +  "/" + sr + "/training_data.txt"
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

def initiate_execution_time() :
    ets = {}

    for sr in SR :
        fpath = "data/" + TTS +  "/" + sr + "/execution_time.txt"
        file = open(fpath)
        lines = file.readlines()
        id = 0
        et = {}
        for l in lines:
            id += 1
            val = l.split(",")[-1][:-1]
            # print(val)
            if (val != "") :
                et[id] = float(val)
        file.close()

        ets[sr] = et

    return ets


def get_generating_time():
    gt = {}

    fpath = "data/" + TTS + "/generating_time.txt"
    file = open(fpath)
    lines = file.readlines()
    id = 0
    for l in lines:
        id += 1
        val = l.split(",")[-1][:-1]
        # print(val)
        if (val != ""):
            gt[id] = float(val)
    file.close()

    return gt

def get_case(cases, id) :
    case = {}
    
    for sr in SR :
        case[sr] = cases[sr][id]

    return case

def get_average_execution_time(ets):
    avg = {}
    for sr in SR :
        avg[sr] = round(np.mean(list(ets[sr].values())), 2)
    return avg

def get_execution_time(ets, avg, id) :
    et = 0.0
    for sr in SR :
        if id in ets[sr].keys() :
            et += ets[sr][id]
        else :
            et += avg[sr]

    return et


def classify_bert(text):
    text = utils.preprocess_text(text)
    resp = requests.get(
        'http://10.4.4.55:5000/translate?text=' + urllib.parse.quote(text))
    if resp.status_code != 200:
        raise 'GET /translate/ {}'.format(resp.status_code)
    return int(resp.content.decode("utf-8"))
    


if __name__ == '__main__' :  

    fpath = "corpus/europarl-20k.txt"
    corpus = get_corpus(fpath)

    test_corpus = []
    train_size = math.ceil(len(corpus) * 3 / 4)
    for i in range(train_size, len(corpus)):
    # for i in range(0, train_size):
        test_corpus.append(corpus[i])

    cases = initiate_cases()
    ets = initiate_execution_time()
    gt = get_generating_time()
    avg = get_average_execution_time(ets)

    # print(get_execution_time(ets, avg, 1))

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
        cumulative_time = 0

        i = 0 # number of texts processed
        while (not q.empty() and last_time <= time_max) :
            i += 1
            data = q.get()           

            is_predicted_bug = classify_bert(data["text"])
            if (is_predicted_bug) :
                case = get_case(cases, data["id"])
                cumulative_time += gt[data["id"]] + get_execution_time(ets, avg, data["id"])
                if constant.UNDETERMINED_TEST_CASE in case.values() :
                    print("Can't determine bug")
                elif constant.FAIL_TEST_CASE not in case.values():
                    print("All Speech Recognition can recognize the speech")
                else :
                    # print("\n\n\n")
                    # print("text")
                    # print(data["text"])
                    # print("cases")
                    # print(case)
                    
                    current_bug += 1
                    
                    bug = {}
                    
                    # bug["number_of_data"] = i
                    bug["number_of_bug"] = list(case.values()).count(constant.FAIL_TEST_CASE)
                    # bug["id_corpus"] = data["id"]
                    for sr in SR :
                        # change the label to make easier in cumulative calculation
                        if case[sr] == -1 :
                            case[sr] = 0

                    bug["case"] = case
                    
                    time_execution = time.time() - start_time + cumulative_time
                    last_time = math.ceil(time_execution / 60.0)
                    bug["time_execution"] = last_time
                    bugs[current_bug] = bug

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
            
                    
                    
        file = open("data/" + TTS + "/result/estimated/with_classifier_" + str(time_max) + "_" +
                    str(datetime.now()) + ".txt", "w+")
        for k in result.keys():
            if k <= time_max :
                file.write("%d, %d, %d, %d, %d, %d\n" % (
                    k, result[k]["number_of_bug"], result[k]["bug_per_asr"][constant.DEEPSPEECH], result[k]["bug_per_asr"][constant.WIT], result[k]["bug_per_asr"][constant.WAV2LETTER], result[k]["bug_per_asr"][constant.PADDLEDEEPSPEECH]))
        
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
        cumulative_time = 0


        i = 0  # number of texts processed
        j = 0  # number of predicted
        while (not q.empty() and last_time <= time_max):
            i += 1
            data = q.get()
            case = get_case(cases, data["id"])
            j += 1
            case = get_case(cases, data["id"])
            cumulative_time += gt[data["id"]] + get_execution_time(ets, avg, data["id"])
            if constant.UNDETERMINED_TEST_CASE in case.values():
                print("Can't determine bug")
            elif constant.FAIL_TEST_CASE not in case.values():
                print("All Speech Recognition can recognize the speech")
            else:
                # print("\n\n\n")
                # print("text")
                # print(data["text"])
                # print("cases")
                # print(case)
                
                current_bug += 1
                time_execution = time.time() - start_time + cumulative_time
                t = math.ceil(time_execution / 60.0)
                bug = {}
                bug["time_execution"] = t
                bug["number_of_bug"] = list(case.values()).count(constant.FAIL_TEST_CASE)
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


        file = open("data/" + TTS + "/result/estimated/without_classifier_" + str(time_max) + "_" +
                    str(datetime.now()) + ".txt", "w+")
        for k in result.keys():
            if k <= time_max:
                file.write("%d, %d, %d, %d, %d, %d\n" % (
                    k, result[k]["number_of_bug"], result[k]["bug_per_asr"][constant.DEEPSPEECH], result[k]["bug_per_asr"][constant.WIT], result[k]["bug_per_asr"][constant.WAV2LETTER], result[k]["bug_per_asr"][constant.PADDLEDEEPSPEECH]))

        file.close()
    

