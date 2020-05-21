import os, sys
import numpy as np
import pandas as pd

from datetime import datetime 

from sentence_splitter import SentenceSplitter, split_text_into_sentences
from nltk.corpus import reuters


splitter = SentenceSplitter(language='en')


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

def get_list_of_sentence_from_reuter_text(paragraph):
    title = paragraph.split("\n")[0]
    bodies = paragraph.split("\n")[1:]
    sentences = splitter.split(text=" ".join(bodies))
    sentences.insert(0, title)
    return sentences


def get_corpus_from_list_of_paragraph(lp):
    s = []
    for paragraph in lp:
        for sentence in paragraph:
            s.append(sentence)
    return s


def generate_reuter_corpus():

    # Extract fileids from the reuters corpus
    fileids = reuters.fileids()

    # Initialize empty lists to store categories and raw text
    categories = []
    text = []

    # Loop through each file id and collect each files categories and raw text
    for file in fileids:
        categories.append(reuters.categories(file))
        text.append(reuters.raw(file))

    # Combine lists into pandas dataframe. reutersDf is the final dataframe.
    reuter_df = pd.DataFrame(
        {'ids': fileids, 'categories': categories, 'text': text})

    list_of_paragraph = reuter_df["text"].apply(
        get_list_of_sentence_from_reuter_text)

    corpus = get_corpus_from_list_of_paragraph(list_of_paragraph)

    df = pd.DataFrame({"sentence": corpus})

    df.to_csv("corpus/reuter-full.csv", index=False)



if __name__ == "__main__" :
    
    # generate_europarl_corpus()
    generate_reuter_corpus()

