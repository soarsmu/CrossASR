import os
import re
import string
from datetime import datetime
import math

import numpy as np
import pandas as pd

from normalise import normalise, tokenize_basic

from sklearn.utils import resample
from g2p_en import G2p

import constant


# read data
def read_data(fpath):
    df = pd.read_csv(fpath, sep=",", header=None)
    df.columns = ["sentence", "label"]
    return df

# shuffle data
def shuffle_data(df):
    return df.sample(frac=1).reset_index(drop=True)

def get_index_in_list_with_have_value(df, value) :
    return np.where(df == value)[0]

# intersection between two sets
def intersection(lst1, lst2):
    lst3 = [value for value in lst1 if value in lst2]
    return lst3

# union between two sets
def union(set1, set2):
    return list(set().union(set1, set2))


def get_fail_test_case(df, column_name="label"):
    bugs = []
    non_bugs = []

    for key in df.keys():
        bugs = union(bugs, get_index_in_list_with_have_value(
            df[key][column_name], constant.FAIL_TEST_CASE))
        non_bugs = union(non_bugs, get_index_in_list_with_have_value(
            df[key][column_name], constant.SUCCESS_TEST_CASE))

    fail_test_case = intersection(bugs, non_bugs)
    
    return fail_test_case

def get_success_test_case(df, column_name="label"):
    bugs = []
    non_bugs = []

    for key in df.keys():
        bugs = union(bugs, get_index_in_list_with_have_value(
            df[key][column_name], constant.FAIL_TEST_CASE))
        non_bugs = union(non_bugs, get_index_in_list_with_have_value(
            df[key][column_name], constant.SUCCESS_TEST_CASE))
    
    success_test_case = []
    for id in non_bugs:
        if id not in bugs:
            success_test_case.append(id)

    return success_test_case


def upsampleMinority(df):
    # Separate majority and minority classes
    df_majority = df[df.label == constant.SUCCESS_TEST_CASE]
    df_minority = df[df.label == constant.FAIL_TEST_CASE]

    # Upsample minority class
    df_minority_upsampled = resample(df_minority,
                                     replace=True,     # sample with replacement
                                     # to match majority class
                                     n_samples=len(df_majority),
                                     random_state=123)  # reproducible results

    # Combine majority class with upsampled minority class
    df_upsampled = pd.concat([df_majority, df_minority_upsampled])

    return df_upsampled.copy()


def downsampleMajority(df):
    # Separate majority and minority classes
    df_majority = df[df.label == constant.SUCCESS_TEST_CASE]
    df_minority = df[df.label == constant.FAIL_TEST_CASE]

    # Downsample majority class
    df_majority_downsampled = resample(df_majority,
                                       replace=True,     # sample with replacement
                                       # to match minority class
                                       n_samples=len(df_minority),
                                       random_state=123)  # reproducible results

    # Combine majority class with upsampled minority class
    df_resampled = pd.concat([df_majority_downsampled, df_minority])

    return df_resampled.copy()


def resampleToFixNumber(df, n):
    # Separate majority and minority classes
    df_non_bug = df[df.label == constant.SUCCESS_TEST_CASE]
    df_bug = df[df.label == constant.FAIL_TEST_CASE]
    df_undetermined = df[df.label == constant.UNDETERMINED_TEST_CASE]

    df_non_bug = resample(df_non_bug,
                          replace=True,     # sample with replacement
                          n_samples=n,    # to match minority class
                          random_state=123)  # reproducible results

    df_bug = resample(df_bug,
                      replace=True,     # sample with replacement
                      n_samples=n,    # to match majority class
                      random_state=123)  # reproducible results

    df_undetermined = resample(df_undetermined,
                               replace=True,     # sample with replacement
                               n_samples=n,    # to match majority class
                               random_state=123)  # reproducible results

    # Combine majority class with upsampled minority class
    df_resampled = pd.concat([df_undetermined, df_bug, df_non_bug])

    return df_resampled.copy()


def getResampleSize(df):
    size = 0
    for k in df.keys():
        df_bug = df[k][df[k].label == constant.FAIL_TEST_CASE]
        if (len(df_bug["label"]) > size):
            size = len(df_bug["label"])

    return size


def resample_to_fix_number(df, n):
    # Separate majority and minority classes
    df_determined = df[df.label == constant.DETERMINED_TEST_CASE]
    df_undetermined = df[df.label == constant.UNDETERMINED_TEST_CASE]

    df_determined = resample(df_determined,
                      replace=True,     # sample with replacement
                      n_samples=n,    # to match majority class
                      random_state=123)  # reproducible results

    df_undetermined = resample(df_undetermined,
                               replace=True,     # sample with replacement
                               n_samples=n,    # to match majority class
                               random_state=123)  # reproducible results

    # Combine majority class with upsampled minority class
    df_resampled = pd.concat([df_undetermined, df_determined])

    return df_resampled.copy()

def get_resample_size(df):
    size = 0
    df_bug = df[df.label == constant.DETERMINED_TEST_CASE]
    if (len(df_bug["label"]) > size):
        size = len(df_bug["label"])

    return size


def remove_punctuation(sentence):
    return sentence.translate(str.maketrans('', '', string.punctuation))

def text_process(sentence):
    nopunc = [char for char in sentence if char not in string.punctuation]
    nopunc = ''.join(nopunc)
    return [word for word in nopunc.split()]


def phoneme_text_process(sentence):
    g2p = G2p()
    nopunc = [char for char in sentence if char not in string.punctuation]
    nopunc = ''.join(nopunc)
    return g2p(nopunc)

def train_classifier(clf, feature_train, labels_train):    
    clf.fit(feature_train, labels_train)
    
def predict_labels(clf, features):
    return clf.predict(features)

def remove_double_space(sentence):
    return re.sub(' +', ' ', sentence)

def remove_punctuation(sentence):
    return sentence.translate(str.maketrans('', '', string.punctuation))

def normalize_text(sentence):
    return " ".join(normalise(sentence, tokenizer=tokenize_basic, verbose=False))

def substitute_word(sentence):
    words = sentence.split(" ")
    preprocessed = []
    for w in words:
        substitution = ""
        if w == "mister":
            substitution = "mr"
        elif w == "missus":
            substitution = "mrs"
        else:
            substitution = w
        preprocessed.append(substitution)
    return " ".join(preprocessed)

def preprocess_text(text):
    text = text.lower()
    text = remove_punctuation(text)
    text = normalize_text(text)
    # need to remove punctuation again as normalise sometimes add punctuation
    text = remove_punctuation(text)
    text = text.lower()
    text = substitute_word(text)
    text = remove_double_space(text)
    text = text.strip()  # remove leading trailing space
    return text
