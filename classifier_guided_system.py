import random
import queue
from datetime import datetime
import time
import os
import subprocess
import string

import math

import numpy as np

import wave
from gtts import gTTS

from alexa_client import AlexaClient
from alexa_client.alexa_client import constants
from alexa_client.alexa_client import helpers

from wit import Wit

import speech_recognition

from gensim.models import Word2Vec


# Speech
import soundfile as sf  # pip install pysoundfile
# pip install python_speech_features
import python_speech_features as speech_lib

from jiwer import wer

import joblib

from datetime import datetime

class WaitTimeoutError(Exception):
    pass
class RequestError(Exception):
    pass
class UnknownValueError(Exception):
    pass


BUG_LABEL = 1
NON_BUG_LABEL = 0
UNDETERMINED_LABEL = -1


GOOGLE_TTS = "google"
APPLE_TTS = "apple"
TTS = GOOGLE_TTS

DEEPSPEECH = "deepspeech"
ALEXA = "alexa"
GCLOUD = "gcloud"
CHROMESPEECH = "gspeech"
WIT = "wit"
WAV2LETTER = "wav2letter"
PADDLEDEEPSPEECH = "paddledeepspeech"
SR = [DEEPSPEECH, WIT, WAV2LETTER, PADDLEDEEPSPEECH]

r = speech_recognition.Recognizer()

CLASSIFIER_MODEL = "SVC"
classifier = {}
for sr in SR :
    classifier_fpath = "model/" + CLASSIFIER_MODEL + "_" + sr + ".sav"
    classifier[sr] = joblib.load(classifier_fpath)

TRANSFORMER_MODEL = "model/transformer.sav"
transformer = joblib.load(TRANSFORMER_MODEL)


# alexa 
client_id = os.getenv("CLIENT_ID")
client_secret = os.getenv("CLIENT_SECRET")
refresh_token = os.getenv("REFRESH_TOKEN")
BASE_URL_NORTH_AMERICA = 'alexa.na.gateway.devices.a2z.com'

# wit 
WIT_AI_KEY = "5PBOPP2VVZM3MJFQOKK57YRG4DFWXIBZ"

if ALEXA in SR :
    client = AlexaClient(
        client_id=client_id,
        secret=client_secret,
        refresh_token=refresh_token,
        base_url=BASE_URL_NORTH_AMERICA
    )

    client.connect()  # authenticate and other handshaking steps


def remove_punctuation(sentence):
    return sentence.translate(str.maketrans('', '', string.punctuation))

def get_timestamp() :
    now = datetime.now()
    date_time = now.strftime("%d_%m_%Y_%H_%M_%S")
    return date_time


def get_corpus(fpath) :
    corpus = []
    file = open(fpath)
    lines = file.readlines()
    id = 1
    for l in lines :
        corpus.append({"id": id, "text": l[:-1]})
        id += 1
    file.close()
    # random.shuffle(corpus)
    return corpus


def classify_text(text) :
    
    # text = remove_punctuation(text)
    # sentence = np.array([text])
    # feature = transformer.transform(sentence)
    feature = extract_feature_using_word2vec(text)
    bug_counter = 0
    non_bug_counter = 0
    for sr in SR :
        prediction = predict_using_classifier(sr, feature)
        if (prediction == BUG_LABEL) :
            bug_counter += 1
        elif (prediction == NON_BUG_LABEL) :
            non_bug_counter += 1
            
    is_bug = False
    if (bug_counter > 0 and non_bug_counter > 0) :
        is_bug = True
    
    # print("\n\n\n\n")
    # print("is bug: ")
    # print(is_bug)
    return is_bug

w2v_model = Word2Vec.load("model/word2vec.model")
def extract_feature_using_word2vec(text) :
    tokenized_sentence = text_process(text)
    encoded_docs = [[w2v_model.wv[word] for word in sentence]
                    for sentence in [tokenized_sentence]]
    padded_docs = create_padding_on_sentence(encoded_docs)
    flatten_array = flatten_docs(padded_docs)
    return flatten_array
    
    
def text_process(sentence):
    nopunc = [char for char in sentence if char not in string.punctuation]
    nopunc = ''.join(nopunc)
    return [word for word in nopunc.split()]


NUM_CORES = 4
EMBEDDING_SIZE = 13
MAX_LENGTH = 75

# because the length of each sentence is various
# and we use non-sequential machine learning model
# we need to make padding for each sentences
def create_padding_on_sentence(encoded_docs):
    padded_posts = []

    for post in encoded_docs:

        # Pad short posts with alternating min/max
        if len(post) < MAX_LENGTH:

            padding_size = MAX_LENGTH - len(post)

            for _ in range(0, padding_size):
                post.append(np.zeros((EMBEDDING_SIZE)))

        # Shorten long posts or those odd number length posts we padded to MAX_LENGTH
        if len(post) > MAX_LENGTH:
            post = post[:MAX_LENGTH]

        # Add the post to our new list of padded posts
        padded_posts.append(post)

    return padded_posts


def flatten_docs(padded_docs):
    flatten = []
    for sentence in padded_docs:
        ps = []
        for word in sentence:
            for feature in word:
                ps.append(feature)
        flatten.append(ps)
    return np.asarray(flatten)


def classify_speech(filepath):

    mfcc_num = 13 # same as defined when experiment
    feature = get_audio_feature_from_file(filepath, mfcc_num)
    bug_counter = 0
    non_bug_counter = 0
    print("feature: " + str(feature))
    for sr in SR:
        prediction = predict_using_classifier(sr, feature)
        print(sr + "-predicted: " + str(prediction))
        if (prediction == BUG_LABEL):
            bug_counter += 1
        elif (prediction == NON_BUG_LABEL):
            non_bug_counter += 1

    is_bug = False
    if (bug_counter > 0 and non_bug_counter > 0):
        is_bug = True

    print("\n\n\n\n")
    print("is bug: ")
    print(is_bug)
    return is_bug


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


def predict_using_classifier(sr, feature) :
    if sr in SR :
        predicted = classifier[sr].predict(feature)
        return predicted[0]
    else :
        return bool(random.getrandbits(1))


def generate_speech(text, timestamp) :
    if (TTS == GOOGLE_TTS) :
        tts = gTTS(text, lang='en-us')
        outfile = "guided_data/mp3/" + "audio_%s.mp3" % timestamp
        wavfile = "guided_data/wav/" + "audio_%s.wav" % timestamp
        tts.save(outfile)
        print(outfile)
        print(wavfile)
        linux_folder = "/home/mhilmiasyrofi/Documents/test-case-generation/"
        if os.path.exists(linux_folder):
            os.system('ffmpeg -i /home/mhilmiasyrofi/Documents/test-case-generation/' + outfile + ' -acodec pcm_s16le -ac 1 -ar 16000 /home/mhilmiasyrofi/Documents/test-case-generation/' + wavfile + ' -y')
        else :
            os.system('ffmpeg -i /Users/mhilmiasyrofi/Documents/test-case-generation/' + outfile + ' -acodec pcm_s16le -ac 1 -ar 16000 /Users/mhilmiasyrofi/Documents/test-case-generation/' + wavfile + ' -y')

    inject_alexa_command(timestamp)

    return wavfile

def recognize_speech(timestamp) :
    transcriptions = {}

    for sr in SR :
        if sr == ALEXA:
            transcriptions[ALEXA] = alexa_recognize(timestamp)
        elif sr == DEEPSPEECH :
            transcriptions[DEEPSPEECH] = deepspeech_recognize(timestamp)
        elif sr == GCLOUD :
            transcriptions[GCLOUD] = gcloud_recognize(timestamp)
        elif sr == CHROMESPEECH:
            transcriptions[CHROMESPEECH] = chromespeech_recognize(timestamp)
        elif sr == WIT :
            transcriptions[WIT] = wit_recognize(timestamp)
        elif sr == WAV2LETTER :
            transcriptions[WAV2LETTER] = wav2letter_recognize(timestamp)
        elif sr == PADDLEDEEPSPEECH :
            transcriptions[PADDLEDEEPSPEECH] = paddledeepspeech_recognize(timestamp)
    
    return transcriptions


def alexa_recognize(timestamp):

    dialog_request_id = helpers.generate_unique_id()

    transcription = ""

    try:
        filename = "audio_%s.wav" % timestamp
        fpath = "guided_data/alexa_wav/" + filename
        audio = open(fpath, 'rb')
        directives = client.send_audio_file(
            audio, dialog_request_id=dialog_request_id)

        success = False
        text = ""
        if directives:
            for j, directive in enumerate(directives):
                if directive.name == 'RenderTemplate':
                    payload = directive.payload
                    if ('textField' in payload.keys()):
                        text = payload['textField']
                        success = True
        else:
            print("Audio " + filename + " - Can't get response")

        if (success):
            transcription = text
        else:
            transcription = ""

    except Exception as e:
        pass

    return transcription


def deepspeech_recognize(timestamp) :
    filename = "audio_%s.wav" % timestamp
    fpath = "guided_data/wav/" + filename
    cmd = 'deepspeech --model sr_deepspeech/deepspeech-0.6.1-models/output_graph.pbmm --lm sr_deepspeech/deepspeech-0.6.1-models/lm.binary --trie sr_deepspeech/deepspeech-0.6.1-models/trie --audio ' + fpath

    proc = subprocess.Popen([cmd], stdout=subprocess.PIPE, shell=True)
    (out, err) = proc.communicate()

    transcription = out.decode("utf-8")
    
    return transcription[:-1]

def wit_recognize(timestamp) :
    filename = "audio_%s.wav" % timestamp
    fpath = "guided_data/wav/" + filename

    transcription = ""

    with open(fpath, 'rb') as audio:
        try:
            wit_client = Wit(WIT_AI_KEY)
            transcription = None
            transcription = wit_client.speech(
                audio, None, {'Content-Type': 'audio/wav'})

            if transcription != None:
                if "_text" in transcription:
                    transcription = str(transcription["_text"])
                else:
                    transcription = ""
            else:
                return  ""
        except Exception as e:
            # print("Could not request results from Wit.ai service; {0}".format(e))
            return ""
    
    return transcription


def paddledeepspeech_recognize(timestamp):
    filename = "audio_%s.wav" % timestamp
    fpath = "/test-case-generation/" + "guided_data/wav/" + filename
    
    cmd = 'docker exec -ti deepspeech2 sh -c "cd DeepSpeech && python infer_one_file.py --filename="' + fpath + '" --mean_std_path="models/librispeech/mean_std.npz" --vocab_path="models/librispeech/vocab.txt" --model_path="models/librispeech" --lang_model_path="models/lm/common_crawl_00.prune01111.trie.klm" --decoding_method="ctc_beam_search" --use_gpu=False --beam_size=500 --num_proc_bsearch=8 --num_conv_layers=2 --num_rnn_layers=3 "'

    proc = subprocess.Popen([cmd], stdout=subprocess.PIPE, shell=True)
    (out, err) = proc.communicate()

    transcription = out.decode("utf-8").split("\n")[-2]

    # print("DeepSpeech2 transcription: %s" % transcription)

    return transcription[:-1]

def chromespeech_recognize(timestamp) :

    filename = "audio_%s.wav" % timestamp
    fpath = "guided_data/wav/" + filename

    # use the audio file as the audio source
    with sr.AudioFile(fpath) as source:
        audio = r.record(source)  # read the entire audio file

    transcription = ""

    # recognize speech using Google Speech Recognition
    try:
        # for testing purposes, we're just using the default API key
        # to use another API key, use `r.recognize_google(audio, key="GOOGLE_SPEECH_RECOGNITION_API_KEY")`
        # instead of `r.recognize_google(audio)`
        transcription = r.recognize_google(audio)
    except sr.UnknownValueError:
        pass
    except sr.RequestError as e:
        pass

    return transcription

def gcloud_recognize(timestamp) :

    filename = "audio_%s.wav" % timestamp
    fpath = "guided_data/wav/" + filename

    transcription = ""
    
    # use the audio file as the audio source
    with sr.AudioFile(fpath) as source:
        audio = r.record(source)  # read the entire audio file

    # recognize speech using Google Cloud Speech
    try:
        transcription = r.recognize_google_cloud(audio)
    except sr.UnknownValueError:
        transcription = ""
    except sr.RequestError as e:
        transcription = ""
        # print("Could not request results from Google Cloud Speech service; {0}".format(e))

    return transcription

def wav2letter_recognize(timestamp) :
    filename = "audio_%s.wav" % timestamp
    fpath = "guided_data/wav/" + filename

    cmd = "docker run --rm -v ~/Documents/test-case-generation/:/root/host/ -it --ipc=host --name w2l-multi-thread -a stdin -a stdout -a stderr wav2letter/wav2letter:inference-latest sh -c  \"/root/wav2letter/build/inference/inference/examples/multithreaded_streaming_asr_example  --input_files_base_path /root/host/sr_wav2letter/model  --input_audio_files /root/host/" + fpath + "  --output_files_base_path /root/host/output/wav2letter/\""

    proc = subprocess.Popen([cmd],
                            stdout=subprocess.PIPE, shell=True)
    (out, err) = proc.communicate()

    transcription = concat_wav2letter_transcription(timestamp)

    return transcription


def concat_wav2letter_transcription(timestamp) :
    filename = "audio_%s.wav.txt" % timestamp
    fpath = "output/wav2letter/" + filename

    file = open(fpath)

    lines = file.readlines()

    transcription = ""

    j = 0
    for line in lines:
        j += 1
        if (j != 1):
            part = line.split(",")[-1]
            if part != "":
                transcription += part[:-1]

    transcription = transcription[:-1]

    file.close()

    return transcription


def inject_alexa_command(timestamp) :
    filename = "audio_%s.wav" % timestamp
    fpath = "guided_data/wav/" + filename
    output_file = "guided_data/alexa_wav/" + filename

    data = []
    skill_executor = wave.open("sr_alexa/skill_executor.wav", 'rb')
    gap = wave.open("sr_alexa/gap.wav", 'rb')
    alexa_instruction = wave.open(fpath, 'rb')
    for j in range(0, skill_executor.getnframes()):
        current_frame = skill_executor.readframes(1)
        data.append([alexa_instruction.getparams(), current_frame])
    for j in range(0, gap.getnframes()):
        current_frame = gap.readframes(1)
        data.append([alexa_instruction.getparams(), current_frame])
        data.append([alexa_instruction.getparams(), current_frame])
    for j in range(0, alexa_instruction.getnframes()):
        current_frame = alexa_instruction.readframes(1)
        data.append([alexa_instruction.getparams(), current_frame])

    skill_executor.close()
    alexa_instruction.close()

    output = wave.open(output_file, 'wb')
    output.setparams(data[0][0])
    for params, frames in data:
        output.writeframes(frames)
    output.close()

def calculate_recognition_error(text, transcriptions) :
    errors = {}

    for sr in SR :
        if sr == ALEXA:
            errors[ALEXA] = round(wer(text, preprocess_transcription(transcriptions[ALEXA])), 2)
        elif sr == DEEPSPEECH :
            errors[DEEPSPEECH] = round(wer(text, preprocess_transcription(transcriptions[DEEPSPEECH])), 2)
        elif sr == GCLOUD :
            errors[GCLOUD] = round(wer(text, preprocess_transcription(transcriptions[GCLOUD])), 2)
        elif sr == CHROMESPEECH :
            errors[CHROMESPEECH] = round(wer(text, preprocess_transcription(transcriptions[CHROMESPEECH])), 2)
        elif sr == WIT :
            errors[WIT] = round(wer(text, preprocess_transcription(transcriptions[WIT])), 2)
        elif sr == WAV2LETTER :
            errors[WAV2LETTER] = round(wer(text, preprocess_transcription(transcriptions[WAV2LETTER])), 2)
        elif sr == PADDLEDEEPSPEECH :
            errors[PADDLEDEEPSPEECH] = round(wer(text, preprocess_transcription(transcriptions[PADDLEDEEPSPEECH])), 2)

    return errors


def preprocess_transcription(transcription):
    transcription = transcription.lower()
    words = transcription.split(" ")
    preprocessed = []
    for w in words:
        # substitution = ""
        # if w == "mister":
        #     substitution = "mr"
        # elif w == "missus":
        #     substitution = "mrs"
        # elif w == "can not":
        #     substitution = "cannot"
        # elif w == "mr.":
        #     substitution = "mr"
        # elif w == "i'm":
        #     substitution = "i am"
        # elif w == "you're":
        #     substitution = "you are"
        if w == "1":
            substitution = "one"
        elif w == "2":
            substitution = "two"
        elif w == "3":
            substitution = "three"
        elif w == "4":
            substitution = "four"
        elif w == "5":
            substitution = "five"
        elif w == "6":
            substitution = "six"
        elif w == "7":
            substitution = "seven"
        elif w == "8":
            substitution = "eight"
        elif w == "9":
            substitution = "nine"
        else:
            substitution = w
        preprocessed.append(substitution)
    return " ".join(preprocessed)


def bug_locator(errors) :
    bugs = {}

    for sr in SR :
        current_sr = sr
        current_error = errors[current_sr]
        bugs[current_sr] = 0
        if current_error != 0 :
            for other_sr in SR :
                if current_sr != other_sr :
                    if errors[other_sr] == 0 :
                        bugs[current_sr] = 1

    return bugs
    

def initiate_folders() :
    folders = []
    main_folder = "guided_data/"
    folders.append(main_folder)
    mp3_google_tts_folder =  main_folder + "mp3/"
    folders.append(mp3_google_tts_folder)
    alexa_data_folder = main_folder + "alexa_wav/"
    folders.append(alexa_data_folder)
    wav_folder = main_folder + "wav/"
    folders.append(wav_folder)

    for folder in folders :
        if not os.path.exists(folder):
            os.makedirs(folder)


if __name__ == '__main__' :    

    needed_bugs_max = 30
    
    x = 0
    while x < 3 : 
        x += 1

        initiate_folders()

        fpath = "corpus-sentence.txt"
        corpus = get_corpus(fpath)

        test_corpus = []
        train_size = math.ceil(len(corpus) * 3 / 4)
        for i in range(train_size, len(corpus)):
            test_corpus.append(corpus[i])
        
        corpus = test_corpus.copy()

        # shuffle the data
        random.shuffle(corpus)

        # using queue to process data one by one
        q = queue.Queue()
        for data in corpus :
            q.put(data)

        detected = []
        
        time_execution_with_classifier = {}
        
        start_time = time.time()
        current_bug = 0
        while (not q.empty() and current_bug < needed_bugs_max) :
            data = q.get()           
            
            timestamp = get_timestamp()
            filepath = generate_speech(data["text"], timestamp)
            is_predicted_bug = classify_speech(filepath)
            if (is_predicted_bug) :

            # is_predicted_bug = classify_text(data["text"])
            # if (is_predicted_bug) :
            #     timestamp = get_timestamp()
            #     generate_speech(data["text"], timestamp)
                transcriptions = recognize_speech(timestamp)
                errors = calculate_recognition_error(data["text"], transcriptions)
                bugs = {}
                if 0 not in errors.values() :
                    print("Can't determine bug")
                elif np.sum(errors.values()) == 0 :
                    print("All Speech Recognition can recognize")
                else :
                    bugs = bug_locator(errors)
                
                if (1 in bugs.values()) :
                    print("\n\n\n")
                    print("text")
                    print(data["text"])
                    print("transcriptions")
                    print(transcriptions)
                    print("error")
                    print(errors)
                    print("bugs")
                    print(bugs)
                    # break
                    current_bug += 1
                    time_execution = round(time.time() - start_time, 2)
                    time_execution_with_classifier[current_bug] = time_execution
        
        file = open("result/with_classifier_" + str(needed_bugs_max) + "_" +
                    str(datetime.now()) + ".txt", "w+")
        for k, v in time_execution_with_classifier.items():
            file.write("%d, %f\n" % (k, v))
        
        file.close()




        # using queue to process data one by one
        q = queue.Queue()
        for data in corpus:
            q.put(data)

        detected = []

        time_execution_without_classifier = {}

        start_time = time.time()
        current_bug = 0
        while (not q.empty() and current_bug < needed_bugs_max):
            data = q.get()
            timestamp = get_timestamp()
            generate_speech(data["text"], timestamp)
            transcriptions = recognize_speech(timestamp)
            errors = calculate_recognition_error(data["text"], transcriptions)
            bugs = {}
            if 0 not in errors.values():
                print("Can't determine bug")
            elif np.sum(errors.values()) == 0:
                print("All Speech Recognition can recognize")
            else:
                bugs = bug_locator(errors)

            if (1 in bugs.values()):
                print("\n\n\n")
                print("text")
                print(data["text"])
                print("transcriptions")
                print(transcriptions)
                print("error")
                print(errors)
                print("bugs")
                print(bugs)
                # break
                current_bug += 1
                time_execution = round(time.time() - start_time, 2)
                time_execution_without_classifier[current_bug] = time_execution
        
        file = open("result/without_classifier_" + str(needed_bugs_max) + "_" +
                    str(datetime.now()) + ".txt", "w+")
        for k, v in time_execution_without_classifier.items():
            file.write("%d, %f\n" % (k, v))
        
        file.close()

    
    # print(corpus)

