import os, sys
import re
import string
from datetime import datetime
import math
import subprocess

import numpy as np
import pandas as pd

from normalise import normalise, tokenize_basic

from gtts import gTTS

from wit import Wit

import constant


WIT_ACCESS_TOKEN = os.getenv("WIT_ACCESS_TOKEN")
wit_client = Wit(WIT_ACCESS_TOKEN)

# read data
def read_data(fpath):
    df = pd.read_csv(fpath, sep=",", header=None)
    df.columns = ["sentence", "label"]
    return df

# shuffle data
def shuffle_data(df):
    return df.sample(frac=1, random_state=constant.INITIAL_SEED).reset_index(drop=True)

def get_index_in_list_with_have_value(df, value) :
    return np.where(df == value)[0]

# intersection between two sets
def intersection(lst1, lst2):
    lst3 = [value for value in lst1 if value in lst2]
    return lst3

# union between two sets
def union(set1, set2):
    return list(set().union(set1, set2))

def remove_punctuation(sentence):
    return sentence.translate(str.maketrans('', '', string.punctuation))

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

def synthesizeSpeech(tts, text, fpath) :
    if fpath[-3:] != "wav" :
        print("File path must be ended with .wav")
        sys.exit()
        
    if tts in constant.TTS :
        if tts == constant.GOOGLE :
            googleSynthesize(text, fpath)
        elif tts == constant.RV :
            responsiveVoiceSynthesize(text, fpath)
        elif tts == constant.FESTIVAL :
            festivalSynthesize(text, fpath)
        elif tts == constant.ESPEAK :
            espeakSynthesize(text, fpath)
        else :
            print("TTS is not detected!")
            sys.exit()
    else :
        print("TTS is not available")
        sys.exit()
            
def googleSynthesize(text, fpath) :
    mp3file = fpath[:-3] + "mp3"
    wavfile = fpath
    googleTTS = gTTS(text, lang='en-us')
    googleTTS.save(mp3file)
    os.system('ffmpeg -i $(pwd)/' + mp3file + ' -acodec pcm_s16le -ac 1 -ar 16000 $(pwd)/' + wavfile + ' -y')
    
    
def responsiveVoiceSynthesize(text, fpath) :
    base_folder = "$(pwd)/"
    mp3file = base_folder + fpath[:-3] + "mp3"
    wavfile = base_folder + fpath

    cmd = "rvtts --voice english_us_male --text \"" + text + "\" -o " + mp3file
    
    os.system(cmd)
    os.system('ffmpeg -i ' + mp3file +
            ' -acodec pcm_s16le -ac 1 -ar 16000 ' + wavfile + ' -y')

    
def festivalSynthesize(text, fpath) :    
    wavfile = "$(pwd)/" + fpath
    cmd = "festival -b \"(utt.save.wave (SayText \\\"" + \
        text + "\\\") \\\"" + wavfile + "\\\" 'riff)\""
    
    os.system(cmd)

def espeakSynthesize(text, fpath) :
    wavfile = "$(pwd)/" + fpath
    cmd = "espeak \"" + text + "\" --stdout > " + wavfile
    # print(cmd)
    os.system(cmd)
    os.system('ffmpeg -i ' + wavfile +
                ' -acodec pcm_s16le -ac 1 -ar 16000 ' + wavfile + ' -y')
    
    
def recognizeSpeech(asr, fpath) :
    if not os.path.exists(fpath) :
        print("Audio file doesn't exist")
        return ""
        
    transcription = ""
    if asr in constant.ASR :
        if asr == constant.DEEPSPEECH :
            transcription = deepspeechRecognize(fpath)
        elif asr == constant.PADDLEDEEPSPEECH :
            transcription = paddledeepspeechRecognize(fpath)
        elif asr == constant.WIT :
            transcription = witRecognize(fpath)
        elif asr == constant.WAV2LETTER :
            transcription = wav2letterRecognize(fpath)
    else :
        print("ASR not available!")
        sys.exit()
        
    return transcription

    
def deepspeechRecognize(fpath):
    cmd = "deepspeech --model models/deepspeech/deepspeech-0.6.1-models/output_graph.pbmm --lm models/deepspeech/deepspeech-0.6.1-models/lm.binary --trie models/deepspeech/deepspeech-0.6.1-models/trie --audio " + fpath

    proc = subprocess.Popen([cmd], stdout=subprocess.PIPE, shell=True)
    (out, err) = proc.communicate()

    transcription = out.decode("utf-8")
#     print("DeepSpeech transcription: %s" % transcription)

    return transcription[:-1]


def paddledeepspeechRecognize(fpath):
    cmd = "docker exec -it deepspeech2 curl http://localhost:5000/transcribe?fpath=" + fpath

    proc = subprocess.Popen([cmd], stdout=subprocess.PIPE, shell=True)
    (out, err) = proc.communicate()

    transcription = out.decode("utf-8").split("\n")[-2]

#     print("DeepSpeech2 transcription: %s" % transcription)

    return transcription[:-1]


def wav2letterRecognize(fpath):
    
    cmd = "docker exec -it wav2letter sh -c \"cat /root/host/" + fpath + " | /root/wav2letter/build/inference/inference/examples/simple_streaming_asr_example --input_files_base_path /root/host/models/wav2letter/\""

    proc = subprocess.Popen([cmd],
                            stdout=subprocess.PIPE, shell=True)
    (out, err) = proc.communicate()
    
    transcription = concatWav2letterTranscription(out)

    return transcription


def concatWav2letterTranscription(out):
    lines = out.splitlines()[21:-2]
#     print(lines)
    transcription = ""

    j = 0
    for line in lines:
        line = line.decode()
        part = line.split(",")[-1]
        if part != "":
            transcription += part

    transcription = transcription[:-1]

    return transcription


def witRecognize(fpath):
    
    transcription = ""
    with open(fpath, 'rb') as audio:
        try:
            transcription = None
            transcription = wit_client.speech(audio, None, {'Content-Type': 'audio/wav'})

            if transcription != None:
                if "text" in transcription:
                    transcription = str(transcription["text"])
                else:
                    return ""
            else:
                return ""
        except Exception as e:
#             print("Could not request results from Wit.ai service; {0}".format(e))
            return ""

    return transcription

def isEmptyFile(fpath) :
    filesize = os.path.getsize(fpath)    
    return filesize == 0
