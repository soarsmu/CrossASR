import sys
import os
from time import time
from operator import itemgetter
import pickle
import pandas as pd
import numpy as np
from argparse import ArgumentParser

import utils, constant

from gensim.models import Word2Vec

from sklearn.base import BaseEstimator
from gensim import corpora

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


df = utils.read_data("training_data/wav2letter.txt")
succes_test_case = np.where(df["label"] == constant.NON_BUG_LABEL)[0]
undetermined_case = np.where(df["label"] == constant.UNDETERMINED_LABEL)[0]

df["label"][succes_test_case] = constant.DETERMINED_LABEL
df["label"][undetermined_case] = constant.UNDETERMINED_LABEL

dataset = df.copy()
RESAMPLE_SIZE = utils.get_resample_size(dataset)
dataset = utils.resample_to_fix_number(dataset, RESAMPLE_SIZE)


class Word2VecModel(BaseEstimator):

    def __init__(self, window=5, size=300, iter=5):
        self.w2v_model_ = None
        self.size = size
        self.window = window
        self.iter = iter
        self.MAX_LENGTH_ = None

    def fit(self, raw_documents, y=None):

        corpus_sentence = pd.read_csv(
            'corpus-sentence.txt', sep=",", header=None)

        corpus_sentence.columns = ["sentence"]
        tokenized_documents = list(
            corpus_sentence['sentence'].apply(utils.text_process))

        # Initialize model
        self.w2v_model_ = Word2Vec(
            sentences=tokenized_documents,
            size=self.size,
            min_count=1,
            window=self.window,
            iter=self.iter,
            seed=1337
        )

        self.MAX_LENGTH_ = self.max_length_sentence_from_corpus(tokenized_documents)

        return self

    def transform(self, tokenized_documents):
        return self.extract_feature(tokenized_documents, self.size)

    def fit_transform(self, raw_documents, y=None):
        self.fit(raw_documents)
        return self.transform(raw_documents)

    def max_length_sentence_from_corpus(self, docs):
        max = 0
        for sentence in docs:
            if (len(sentence) > max):
                max = len(sentence)
        return max
    
    # because the length of each sentence is various
    # and we use non-sequential machine learning model
    # we need to make padding for each sentences
    def create_padding_on_sentence(self, encoded_docs, embedding_size):
        padded_posts = []

        for post in encoded_docs:

            # Pad short posts with alternating min/max
            if len(post) < self.MAX_LENGTH_:

                padding_size = self.MAX_LENGTH_ - len(post)

                for _ in range(0, padding_size):
                    post.append(np.zeros((embedding_size)))

            # Shorten long posts or those odd number length posts we padded to MAX_LENGTH
            if len(post) > self.MAX_LENGTH_:
                post = post[:self.MAX_LENGTH_]

            # Add the post to our new list of padded posts
            padded_posts.append(post)

        return padded_posts

    def flatten_docs(self, padded_docs):
        flatten = []
        for sentence in padded_docs:
            ps = []
            for word in sentence:
                for feature in word:
                    ps.append(feature)
            flatten.append(ps)
        return np.asarray(flatten)

    def extract_feature(self, docs, embedding_size):
        tokenized_sentences = docs.apply(utils.text_process)
        encoded_docs = [[self.w2v_model_.wv[word] for word in sentence]
                        for sentence in tokenized_sentences]
        padded_docs = self.create_padding_on_sentence(encoded_docs, embedding_size)
        flatten_array = self.flatten_docs(padded_docs)
        return flatten_array


param_grid = {'word2vec__size': (10, 13, 15, 30, 50, 100, 200, 300),
              'word2vec__iter': (5, 10, 20)
              }

pipe_log = Pipeline([('word2vec', Word2VecModel()),
                     ('classifier', RandomForestClassifier(n_estimators=31, random_state=111))])

log_grid = GridSearchCV(pipe_log,
                        param_grid=param_grid,
                        scoring="accuracy",
                        verbose=3,
                        n_jobs=4)

fitted = log_grid.fit(dataset["sentence"], dataset["label"])

# Best parameters
print("Best Parameters: {}\n".format(log_grid.best_params_))
print("Best accuracy: {}\n".format(log_grid.best_score_))
print("Finished.")

