# Basic libraries
import scipy.sparse as sparse
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.utils import resample
from sklearn.base import clone as clone_model
import nltk

# pip install python_speech_features
import python_speech_features as speech_lib
import soundfile as sf  # pip install pysoundfile
import os
import sys
import numpy as np
import re
import string
from datetime import datetime
import math
import pandas as pd

import utils


DEEPSPEECH = "deepspeech"
ALEXA = "alexa"
GCLOUD = "gcloud"
CHROMESPEECH = "gspeech"
WIT = "wit"
WAV2LETTER = "wav2letter"
PADDLEDEEPSPEECH = "paddledeepspeech"
SR = [DEEPSPEECH, WIT, WAV2LETTER, PADDLEDEEPSPEECH]


def get_audio_feature_from_file(filepath, mfcc_num):
    feature = None
    with open(filepath, 'rb') as f:

        signal, samplerate = sf.read(f)

        MFCCs = speech_lib.mfcc(signal, samplerate, winlen=0.060, winstep=0.03, numcep=mfcc_num,
                                nfilt=mfcc_num, nfft=960, lowfreq=0, highfreq=None, preemph=0.97,
                                ceplifter=22, appendEnergy=False)

        #mean value of each MFCC over the sample
        mean_mfcc = np.expand_dims(np.mean(MFCCs, axis=0), axis=1).T
        feature = mean_mfcc

    return feature


def create_audio_features(corpus_filepath):
    file = open(corpus_filepath)
    corpus = file.readlines()
    file.close()

    mfcc_num = 13
    dataset = np.ndarray(shape=(0, mfcc_num))
    labels = np.ndarray(shape=(0, 1))

    audio_folder = "./data/tts_google/generated_speech/"

    i = 0
    for sentence in corpus:
        i += 1
        filepath = os.path.join(audio_folder, "audio_" + str(int(i)) + ".wav")

        mean_mfcc = get_audio_feature_from_file(filepath, mfcc_num)
        if (mean_mfcc.any() != None):
            dataset = np.append(dataset, mean_mfcc, axis=0)

    # reformat to Panda DataFrame
    arr = sparse.coo_matrix(dataset)
    return arr.toarray().tolist()


if __name__ == "__main__" :

    BUG_LABEL = 1
    NON_BUG_LABEL = 0
    UNDETERMINED_LABEL = -1

    audio_features = create_audio_features("corpus-sentence.txt")


    N_JOBS = 4

    svc = SVC(kernel='sigmoid', gamma=1.0)
    knc = KNeighborsClassifier(n_neighbors=10)
    dtc = DecisionTreeClassifier(min_samples_split=7, random_state=111)
    lrc = LogisticRegression(solver='lbfgs', n_jobs=N_JOBS)
    rfc = RandomForestClassifier(n_estimators=31, random_state=111, n_jobs=N_JOBS)
    abc = AdaBoostClassifier(n_estimators=31, random_state=111)
    bc = BaggingClassifier(n_estimators=10, random_state=111, n_jobs=N_JOBS)
    etc = ExtraTreesClassifier(n_estimators=10, random_state=111, n_jobs=N_JOBS)

    clfs = {'SVC' : svc,'KN' : knc, 'DT': dtc, 'LR': lrc, 'RF': rfc, 'AdaBoost': abc, 'BgC': bc, 'ETC': etc }

    x = 0
    while x < 3:
        x += 1

        df = {}
        for sr in SR:
            df[sr] = utils.read_data("training_data/" + sr + ".txt")
        
        for key in df.keys():
            df[key] = df[key].drop(columns=["sentence"])
            df[key]["audio_feature"] = audio_features

        RESAMPLE_SIZE = utils.getResampleSize(df)

        # shuffle data
        for key in df.keys():
            df[key] = utils.shuffle_data(df[key])
        
        k = list(df.keys())[0]
        FIRST_BATCH_SIZE = math.ceil(len(df[k]["label"]) / 2)
        BATCH_SIZE = math.ceil(len(df[k]["label"]) / 2 / 3)
        MAX_SIZE = len(df[k]["label"])

        features = {}
        labels = {}
        for k, v in df.items():
            data = v[:FIRST_BATCH_SIZE].copy()
            data = utils.resampleToFixNumber(data, RESAMPLE_SIZE)
            features[k] = list(data["audio_feature"])
            labels[k] = data["label"]

        models = {}
        selected_clfs = ["KN", "DT", "LR", "RF", "AdaBoost", "BgC", "ETC"]
        for k in selected_clfs:
            per_sr_model = {}
            for key_sr in features.keys():
                model = None
                model = clone_model(clfs[k])
                utils.train_classifier(model, features[key_sr], labels[key_sr])
                per_sr_model[key_sr] = model
            models[k] = per_sr_model

        performance_writer = open(
            "performance/audio_feature_" + str(datetime.now()) + ".txt", "w+")

        for key_clf, v in models.items():
            # classify the selected data
            training_data = {}
            for key_sr, model in v.items():
                training_data[key_sr] = df[key_sr][:FIRST_BATCH_SIZE].copy()

            lower_bound = FIRST_BATCH_SIZE

            columns = ["True", "Positive", "TP", "Precision", "Recall", "F1"]
            index = []
            performance_data = []

            while (lower_bound < MAX_SIZE):
                # set the upper bound size
                upper_bound = lower_bound + BATCH_SIZE
                if (upper_bound > MAX_SIZE):
                    upper_bound = MAX_SIZE
                batch = str(lower_bound) + " - " + str(upper_bound)
                index.append(batch)
                
                df_per_batch = {}

                for key_sr, model in v.items():
                    # get the data to be selected
                    current_data = df[key_sr][lower_bound:upper_bound].copy()

                    # get the feature and label
                    current_features = list(current_data["audio_feature"])
                    current_labels = current_data["label"]

                    # predict the current data
                    pred = utils.predict_labels(model, current_features)

                    d = {
                        "label" : current_data["label"].values,
                        "prediction" : pred
                        }
        
                    df_per_batch[key_sr] = pd.DataFrame(data=d)


                predicted_fail_test_case = utils.get_fail_test_case(df_per_batch, "prediction")
                actual_fail_test_case = utils.get_fail_test_case(df_per_batch, "label")
                
                correctly_predicted_fail_test_case = utils.intersection(predicted_fail_test_case, actual_fail_test_case)
                
                
                precision = 0
                if (len(predicted_fail_test_case) != 0) :
                    precision = round(len(correctly_predicted_fail_test_case)/len(predicted_fail_test_case), 2)
                recall = 0
                if (len(actual_fail_test_case) != 0) :
                    recall = round(len(correctly_predicted_fail_test_case)/len(actual_fail_test_case), 2)
                precision_add_recall = precision + recall
                f1 = 0
                if precision_add_recall != 0 :
                    f1 = round( 2 * precision * recall / (precision_add_recall), 2)
                performance_data.append(
                    [len(actual_fail_test_case), 
                    len(predicted_fail_test_case), 
                    len(correctly_predicted_fail_test_case), 
                    precision, recall, f1])

                # add the predicted data to previous data
                if (len(correctly_predicted_fail_test_case) > 0):
                    for key_sr in df_per_batch.keys():
                        added_data = current_data.iloc[correctly_predicted_fail_test_case, :]
                        training_data[key_sr].append(added_data)

                        # handle imbalance data
                        resampled_data = utils.resampleToFixNumber(
                            training_data[key_sr], RESAMPLE_SIZE)

                        # re-extract feature
                        features[key_sr] = list(resampled_data["audio_feature"])
                        labels[key_sr] = resampled_data["label"]

                        # re-train the model
                        utils.train_classifier(model, features[key_sr], labels[key_sr])

                # update the lower bound
                lower_bound = upper_bound

            performance = pd.DataFrame(performance_data, index=index, columns=columns)

            print("Classifier: " + key_clf)
            print(performance)
            actual_fail_test_case = int(performance["True"].sum(axis=0))
            predicted_fail_test_case = int(performance["Positive"].sum(axis=0))
            correctly_predicted_fail_test_case = int(performance["TP"].sum(axis=0))
            precision = round(performance["Precision"].mean(), 2)
            recall = round(performance["Recall"].mean(), 2)
            f1 = round(performance["F1"].mean(), 2)

            print("Actual Fail Test Case: %d" % (actual_fail_test_case))
            print("Predicted Fail Test Case: %d" % (predicted_fail_test_case))
            print("Correctly Predicted Fail Test Case: %d" %
                (correctly_predicted_fail_test_case))
            print("Precision: %.2f" % (precision))
            print("Recall: %.2f" % (recall))
            print("F1: %.2f" % (f1))
            print("\n")

            performance_writer.write("Classifier: %s" % (key_clf))
            performance_writer.write("\n" + str(performance))
            performance_writer.write("\nActual Fail Test Case: %d" %
                                    (actual_fail_test_case))
            performance_writer.write(
                "\nPredicted Fail Test Case: %d" % (predicted_fail_test_case))
            performance_writer.write("\nCorrectly Predicted Fail Test Case: %d" % (
                correctly_predicted_fail_test_case))
            performance_writer.write("\nPrecision: %.2f" % (precision))
            performance_writer.write("\nRecall: %.2f" % (recall))
            performance_writer.write("\nF1: %.2f" % (f1))
            performance_writer.write("\n\n")


        performance_writer.close()
