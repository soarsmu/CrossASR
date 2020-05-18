import random
import queue
from datetime import datetime
import time
import os
import subprocess
import string

import math

import constant, utils

import requests
import urllib

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


SR = [constant.DEEPSPEECH, constant.WIT, constant.WAV2LETTER, constant.PADDLEDEEPSPEECH]
TTS = constant.ESPEAK

r = speech_recognition.Recognizer()

# alexa
client_id = os.getenv("CLIENT_ID")
client_secret = os.getenv("CLIENT_SECRET")
refresh_token = os.getenv("REFRESH_TOKEN")
BASE_URL_NORTH_AMERICA = 'alexa.na.gateway.devices.a2z.com'

# wit
WIT_AI_KEY = "5PBOPP2VVZM3MJFQOKK57YRG4DFWXIBZ"

if constant.ALEXA in SR:
    client = AlexaClient(
        client_id=client_id,
        secret=client_secret,
        refresh_token=refresh_token,
        base_url=BASE_URL_NORTH_AMERICA
    )

    client.connect()  # authenticate and other handshaking steps


def remove_punctuation(sentence):
    return sentence.translate(str.maketrans('', '', string.punctuation))


def get_timestamp():
    now = datetime.now()
    date_time = now.strftime("%d_%m_%Y_%H_%M_%S")
    return date_time


def get_corpus(fpath):
    corpus = []
    file = open(fpath)
    lines = file.readlines()
    id = 1
    for l in lines:
        corpus.append({"id": id, "text": l[:-1]})
        id += 1
    file.close()
    # random.shuffle(corpus)
    return corpus


def classify_bert(text):
    resp = requests.get(
        'http://10.4.4.55:5000/translate?text=' + urllib.parse.quote(text))
    if resp.status_code != 200:
        raise 'GET /translate/ {}'.format(resp.status_code)
    return int(resp.content.decode("utf-8"))


def classify_text(text):
    return classify_bert(text)


def text_process(sentence):
    nopunc = [char for char in sentence if char not in string.punctuation]
    nopunc = ''.join(nopunc)
    return [word for word in nopunc.split()]


def generate_speech(text, timestamp):
    wavfile = "guided_data/wav/" + "audio_%s.wav" % timestamp

    if (TTS == constant.GOOGLE):
        tts = gTTS(text, lang='en-us')
        outfile = "guided_data/mp3/" + "audio_%s.mp3" % timestamp
        tts.save(outfile)
        
        os.system('ffmpeg -i ' + constant.BASE_FOLDER + outfile +
                    ' -acodec pcm_s16le -ac 1 -ar 16000 ' + constant.BASE_FOLDER + wavfile + ' -y')

        inject_alexa_command(timestamp)
    
    elif (TTS == constant.FESTIVAL):
        cmd = "festival -b \"(utt.save.wave (SayText \\\"" + \
            text + "\\\") \\\"" + constant.BASE_FOLDER + \
            wavfile + "\\\" 'riff)\""
        # print(cmd)
        os.system(cmd)
        # print("finish-smd")
    elif (TTS == constant.RV) :
        mp3_folder = "tts_rv/mp3_generated_speech/"

        outfile = constant.BASE_FOLDER + mp3_folder + "audio_%s.mp3" % timestamp

        cmd = "rvtts --voice english_us_male --text \"" + text + "\" -o " + outfile
        # print(cmd)
        os.system(cmd)
        os.system('ffmpeg -i ' + outfile +
                ' -acodec pcm_s16le -ac 1 -ar 16000 ' + constant.BASE_FOLDER + wavfile + ' -y')
    elif (TTS == constant.ESPEAK):
        cmd = "espeak \"" + text + "\" --stdout > " + \
            constant.BASE_FOLDER + wavfile
        os.system(cmd)

        outfile = constant.BASE_FOLDER + "tts_espeak/temp_data/" + \
            "audio_" + str(timestamp) + ".wav"
        wavfile = "/home/mhilmiasyrofi/Documents/cross-asr/" + \
            wavfile
        cmd = "espeak \"" + text + "\" --stdout > " + outfile
        # print(cmd)
        os.system(cmd)
        os.system('ffmpeg -i ' + outfile +
                  ' -acodec pcm_s16le -ac 1 -ar 16000 ' + wavfile + ' -y')


    
    return wavfile


def recognize_speech(timestamp):
    transcriptions = {}

    for sr in SR:
        if sr == constant.ALEXA:
            transcriptions[constant.ALEXA] = alexa_recognize(timestamp)
        elif sr == constant.DEEPSPEECH:
            transcriptions[constant.DEEPSPEECH] = deepspeech_recognize(timestamp)
        elif sr == constant.GCLOUD:
            transcriptions[constant.GCLOUD] = gcloud_recognize(timestamp)
        elif sr == constant.CHROMESPEECH:
            transcriptions[constant.CHROMESPEECH] = chromespeech_recognize(timestamp)
        elif sr == constant.WIT:
            transcriptions[constant.WIT] = wit_recognize(timestamp)
        elif sr == constant.WAV2LETTER:
            transcriptions[constant.WAV2LETTER] = wav2letter_recognize(timestamp)
        elif sr == constant.PADDLEDEEPSPEECH:
            transcriptions[constant.PADDLEDEEPSPEECH] = paddledeepspeech_recognize(
                timestamp)

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


def deepspeech_recognize(timestamp):
    filename = "audio_%s.wav" % timestamp
    fpath = "guided_data/wav/" + filename
    cmd = 'deepspeech --model sr_deepspeech/deepspeech-0.6.1-models/output_graph.pbmm --lm sr_deepspeech/deepspeech-0.6.1-models/lm.binary --trie sr_deepspeech/deepspeech-0.6.1-models/trie --audio ' + fpath

    proc = subprocess.Popen([cmd], stdout=subprocess.PIPE, shell=True)
    (out, err) = proc.communicate()

    transcription = out.decode("utf-8")

    return transcription[:-1]


def wit_recognize(timestamp):
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
                return ""
        except Exception as e:
            # print("Could not request results from Wit.ai service; {0}".format(e))
            return ""

    return transcription


def paddledeepspeech_recognize(timestamp):
    filename = "audio_%s.wav" % timestamp
    fpath = "/cross-asr/" + "guided_data/wav/" + filename

    cmd = 'docker exec -ti deepspeech2 sh -c "cd DeepSpeech && python infer_one_file.py --filename="' + fpath + \
        '" --mean_std_path="models/librispeech/mean_std.npz" --vocab_path="models/librispeech/vocab.txt" --model_path="models/librispeech" --lang_model_path="models/lm/common_crawl_00.prune01111.trie.klm" --decoding_method="ctc_beam_search" --use_gpu=False --beam_size=500 --num_proc_bsearch=8 --num_conv_layers=2 --num_rnn_layers=3 "'

    proc = subprocess.Popen([cmd], stdout=subprocess.PIPE, shell=True)
    (out, err) = proc.communicate()

    transcription = out.decode("utf-8").split("\n")[-2]

    # print("DeepSpeech2 transcription: %s" % transcription)

    return transcription[:-1]


def chromespeech_recognize(timestamp):

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


def gcloud_recognize(timestamp):

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


def wav2letter_recognize(timestamp):
    filename = "audio_%s.wav" % timestamp
    fpath = "guided_data/wav/" + filename

    cmd = "docker run --rm -v " + constant.BASE_FOLDER + ":/root/host/ -it --ipc=host --name w2l-multi-thread -a stdin -a stdout -a stderr wav2letter/wav2letter:inference-latest sh -c  \"/root/wav2letter/build/inference/inference/examples/multithreaded_streaming_asr_example  --input_files_base_path /root/host/sr_wav2letter/model  --input_audio_files /root/host/" + fpath + "  --output_files_base_path /root/host/output/wav2letter/\""

    proc = subprocess.Popen([cmd],
                            stdout=subprocess.PIPE, shell=True)
    (out, err) = proc.communicate()

    transcription = concat_wav2letter_transcription(timestamp)

    return transcription


def concat_wav2letter_transcription(timestamp):
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


def inject_alexa_command(timestamp):
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


def calculate_recognition_error(text, transcriptions):
    errors = {}

    for sr in SR:
        if sr == constant.ALEXA:
            errors[constant.ALEXA] = round(
                wer(text, utils.preprocess_text(transcriptions[constant.ALEXA])), 2)
        elif sr == constant.DEEPSPEECH:
            errors[constant.DEEPSPEECH] = round(
                wer(text, utils.preprocess_text(transcriptions[constant.DEEPSPEECH])), 2)
        elif sr == constant.GCLOUD:
            errors[constant.GCLOUD] = round(
                wer(text, utils.preprocess_text(transcriptions[constant.GCLOUD])), 2)
        elif sr == constant.CHROMESPEECH:
            errors[constant.CHROMESPEECH] = round(
                wer(text, utils.preprocess_text(transcriptions[constant.CHROMESPEECH])), 2)
        elif sr == constant.WIT:
            errors[constant.WIT] = round(
                wer(text, utils.preprocess_text(transcriptions[constant.WIT])), 2)
        elif sr == constant.WAV2LETTER:
            errors[constant.WAV2LETTER] = round(
                wer(text, utils.preprocess_text(transcriptions[constant.WAV2LETTER])), 2)
        elif sr == constant.PADDLEDEEPSPEECH:
            errors[constant.PADDLEDEEPSPEECH] = round(
                wer(text, utils.preprocess_text(transcriptions[constant.PADDLEDEEPSPEECH])), 2)

    return errors

def bug_locator(errors):
    bugs = {}

    for sr in SR:
        current_sr = sr
        current_error = errors[current_sr]
        bugs[current_sr] = 0
        if current_error != 0:
            for other_sr in SR:
                if current_sr != other_sr:
                    if errors[other_sr] == 0:
                        bugs[current_sr] = 1

    return bugs


def initiate_folders():
    folders = []
    main_folder = "guided_data/"
    folders.append(main_folder)
    mp3_google_tts_folder = main_folder + "mp3/"
    folders.append(mp3_google_tts_folder)
    alexa_data_folder = main_folder + "alexa_wav/"
    folders.append(alexa_data_folder)
    wav_folder = main_folder + "wav/"
    folders.append(wav_folder)

    for folder in folders:
        if not os.path.exists(folder):
            os.makedirs(folder)

def is_directory_empty(directory):
    return not bool(directory)


if __name__ == '__main__':

    time_max = 6 # in minutes

    x = 0
    while x < 1:
        x += 1

        initiate_folders()

        fpath = "corpus/europarl.txt"
        corpus = get_corpus(fpath)

        test_corpus = []
        train_size = math.ceil(len(corpus) * 3 / 4)
        for i in range(len(corpus)):
        # for i in range(train_size, len(corpus)):
            test_corpus.append(corpus[i])

        corpus = test_corpus.copy()

        # shuffle the data
        random.seed(100 + x)
        random.shuffle(corpus)

        # using queue to process data one by one
        q = queue.Queue()
        for data in corpus:
            q.put(data)

        bugs = {}

        start_time = time.time()
        current_bug = 0
        last_time = 0

        i = 0  # number of texts processed
        while (not q.empty() and last_time <= time_max):
            i += 1
            data = q.get()
            timestamp = get_timestamp()
            generate_speech(data["text"], timestamp)
            transcriptions = recognize_speech(timestamp)
            errors = calculate_recognition_error(data["text"], transcriptions)
            case = {}
            if 0 not in errors.values():
                # print("\n\n\n")
                print("Can't determine bug")
                # print(errors)
            elif np.sum(errors.values()) == 0:
                print("All Speech Recognition can recognize")
            else:
                case = bug_locator(errors)
            
            time_execution = time.time() - start_time
            last_time = math.ceil(time_execution / 60.0)
            
            if (1 in case.values()):
                print("\n\n\n")
                print("text")
                print(data["text"])
                print("transcriptions")
                print(transcriptions)
                print("error")
                print(errors)
                print("bugs")
                print(case)

                current_bug += 1

                bug = {}

                bug["time_execution"] = last_time
                bug["number_of_bug"] = list(case.values()).count(constant.FAIL_TEST_CASE)
                # bug["id_corpus"] = data["id"]
                
                bug["case"] = case

                bugs[current_bug] = bug


        file = open("result/real/without_classifier_" + str(time_max) + "_" +
                    str(datetime.now()) + ".txt", "w+")

        if not is_directory_empty(bug) :

            result = {}

            last_time = 0
            last_total_bug = 0
            last_bugs = {}
            for sr in SR:
                last_bugs[sr] = 0

            for k, v in bugs.items():
                for i in range(last_time, v["time_execution"]):
                    result[i+1] = {"number_of_bug": last_total_bug,
                                "bug_per_asr": last_bugs.copy()}
                last_total_bug += v["number_of_bug"]
                for sr in SR:
                    # cumulative calculation
                    last_bugs[sr] += v["case"][sr]
                result[v["time_execution"]] = {
                    "number_of_bug": last_total_bug, "bug_per_asr": last_bugs.copy()}
                last_time = v["time_execution"]


            for k in result.keys():
                if k <= time_max:
                    file.write("%d, %d, %d, %d, %d, %d\n" % (
                        k, result[k]["number_of_bug"], result[k]["bug_per_asr"][constant.DEEPSPEECH], result[k]["bug_per_asr"][constant.WIT], result[k]["bug_per_asr"][constant.WAV2LETTER], result[k]["bug_per_asr"][constant.PADDLEDEEPSPEECH]))

        else :
            file.write("No fail test case found!")
        
        file.close()
