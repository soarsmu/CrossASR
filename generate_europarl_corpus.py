import os, sys
import numpy as np
import pandas as pd

from normalise import normalise, tokenize_basic
import string
import re

from datetime import datetime 


def remove_double_space(sentence):
    return re.sub(' +', ' ', sentence)


def remove_punctuation(sentence):
    return sentence.translate(str.maketrans('', '', string.punctuation))


def normalize_text(sentence):
    return " ".join(normalise(sentence, tokenizer=tokenize_basic, verbose=False))


def clean_text(text):
    text = text.lower()
    text = remove_punctuation(text)
    text = normalize_text(text)
    text = text.lower()
    text = remove_double_space(text)
    text = text.strip()  # remove leading trailing space
    return text


def concat_data() :

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
                
    df.to_csv("corpus/europarl-full.csv", index=False)

def get_sample_data(n):
    
    print("read data: " + str(datetime.now()))
    fpath = "corpus/europarl-full.csv"
    df = pd.read_csv(fpath, delimiter=',')
    df = df.dropna()
    df = df.drop_duplicates()
    df = df.reset_index(drop=True)

    print("random: " + str(datetime.now()))
    seed = 12345
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    sample = df["English"][:n]

    return pd.DataFrame(sample, columns=["English"])

def preprocess_data(df, n) :
    print("cleaning: " + str(datetime.now()))
    clean_df = df["English"].apply(clean_text)

    # remove empty string
    clean_df = [i for i in clean_df if i]

    file = open("corpus/europarl.txt", "w+")

    for s in clean_df[:n]:
        file.write("%s\n" % s)

    file.close()

    print("finish: " + str(datetime.now()))



if __name__ == "__main__" :
    # concat_data()
    print("start: " + str(datetime.now()))
    n = 20000
    df = get_sample_data(int(n*1.1))
#     preprocess_data(df, n)

