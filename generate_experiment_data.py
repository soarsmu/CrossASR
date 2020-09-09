import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime

import utils, constant 

def generate_europarl_corpus() :

    # download full data from https://www.kaggle.com/djonafegnem/europarl-parallel-corpus-19962011
    data = "europarl-parallel-corpus-19962011/"

    df = pd.DataFrame([""], columns=["English"])
    for (dirpath, _, filenames) in os.walk(data):
        print(filenames)
        for f in filenames :
            if ".csv" in f :
                fpath = data + f
                print(fpath)
                d = pd.read_csv(fpath, delimiter=',')
                df = pd.concat([df["English"], d["English"]])
                df = pd.DataFrame(df, columns=["English"])

    df = df.rename(columns={'English': 'sentence'})
                
    df.to_csv("corpus/europarl-full.csv", index=False)

    
def get_sample_data(df, n):
    seed = constant.INITIAL_SEED
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    sample = df[:n]

    return pd.DataFrame(sample, columns=["sentence"])


def preprocess_data(df, n):
    
    clean_df = df["sentence"].apply(utils.preprocess_text)

    # remove empty string
    clean_df = [i for i in clean_df if i]
    
    return clean_df[:n]

if __name__ == "__main__":

    print("start: " + str(datetime.now()))
    
    generate_europarl_corpus()

    print("generate corpus: " + str(datetime.now()))
    
    DATASET = constant.DATASET
    
    n = 20000
    fpath = "corpus/europarl-full.csv"

    print("read data: " + str(datetime.now()))
    
    df = pd.read_csv(fpath, delimiter=',')
    
    # drop null
    df = df.dropna()
    
    # drop duplicates
    df = df.drop_duplicates()
    
    # reset index
    df = df.reset_index(drop=True)
    
    print("get sample: " + str(datetime.now()))
    # get sample data
    sample_df = get_sample_data(df, int(n*1.1))
    
    print("preprocess data: " + str(datetime.now()))
    # text preprocessing
    data = preprocess_data(sample_df, n)
    
    
    print("write data: " + str(datetime.now()))
    
    outfile = "corpus/europarl-20k.txt"
    
    file = open(outfile, "w+")
    for s in data:
        file.write("%s\n" % s)
    file.close()

    
    
