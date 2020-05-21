import os
import sys
import numpy as np
import pandas as pd

import utils, constant 

from datetime import datetime

def get_sample_data(df, n):

    print("random: " + str(datetime.now()))
    seed = constant.INITIAL_SEED
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    sample = df[:n]

    return pd.DataFrame(sample, columns=["sentence"])


def preprocess_data(df, n, outfile):
    
    print("cleaning: " + str(datetime.now()))
    clean_df = df["sentence"].apply(utils.preprocess_text)

    # remove empty string
    clean_df = [i for i in clean_df if i]

    file = open(outfile, "w+")

    for s in clean_df[:n]:
        file.write("%s\n" % s)

    file.close()

    print("finish: " + str(datetime.now()))


if __name__ == "__main__":
    
    print("start: " + str(datetime.now()))

    DATASET = constant.DATASET
    
    n = 20000
    fpath = "corpus/" + DATASET + "-full.csv"

    print("read data: " + str(datetime.now()))
    
    df = pd.read_csv(fpath, delimiter=',')
    
    # drop null
    df = df.dropna()
    
    # drop duplicates
    df = df.drop_duplicates()
    
    # reset index
    df = df.reset_index(drop=True)
    
    # get sample data
    sample_df = get_sample_data(df, int(n*1.1))
    
    # text preprocessing
    outfile = "corpus/" + DATASET + "-20k.txt"
    preprocess_data(sample_df, n, outfile)
