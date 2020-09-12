import os
import numpy as np
import pandas as pd
import constant

RANDOM_SEED = [12346, 12347, 12348]
ASR = constant.ASR
TTS = constant.TTS
DATASET = constant.DATASET

import sys, getopt
import utils
import constant

def printHelp() :
    print('calculate_averaged_result.py -a <approach>')
    print("or")
    print('calculate_averaged_result.py --approach <approach>')

def main(argv):
    approach = ""
    try:
        opts, args = getopt.getopt(argv,"ha:",["approach="])
    except getopt.GetoptError:
        printHelp()
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            printHelp()
            sys.exit()
        elif opt in ("-a", "--approach"):
            approach = arg
        
    if approach != "" :
        calculateAveragedResult(approach)
    else :
        print("Please specify the output file location")

def calculateAveragedResult(approach) :

    df = {}
    avg = {}

    for tts in TTS:

        a = {}
        avg[tts] = {}

        for sr in ASR:

            b = {}

            avg[tts][sr] = {}

            for random_seed in RANDOM_SEED:

                folder = "result/%s/average/%s/%s/" % (approach, tts, sr)
                if not os.path.exists(folder):
                    os.makedirs(folder)

                fpath = "result/%s/%s-%d/%s/%s/statistic.csv" % (approach,
                                                                DATASET, random_seed, tts, sr)

                b[random_seed] = pd.read_csv(fpath)

            a[sr] = b

        df[tts] = a

    avg = {}
    for tts in TTS:
        t = {}
        for sr in ASR:
            s = {}
            i = RANDOM_SEED[0]
            first = i
            temp = df[tts][sr][i]
            for i in RANDOM_SEED:
                if i != first:
                    temp = temp.add(df[tts][sr][i], fill_value=0)
            t[sr] = temp/len(RANDOM_SEED)
            t[sr] = t[sr].drop(columns=["stc", "utc"])
        avg[tts] = t

    for tts in TTS:
        for sr in ASR:
            fpath = "result/%s/%s-averaged/%s/%s/statistic.csv" % (approach, DATASET,tts, sr)
            avg[tts][sr].to_csv(fpath, index=False, float_format='%.2f')


if __name__ == "__main__":
    main(sys.argv[1:])