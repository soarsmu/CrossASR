import os
import time
from datetime import datetime

wav_folder = "data/tts_espeak/generated_speech/"

if not os.path.exists(wav_folder):
    os.makedirs(wav_folder)

file = open("corpus/europarl.txt")
lines = file.readlines()
file.close()

exec_time_folder = "tts_espeak/execution_time/"
if not os.path.exists(exec_time_folder):
    os.makedirs(exec_time_folder)

file = open(exec_time_folder + str(datetime.now()) + ".txt", "w+")

i = 0
for line in lines:
    i = i + 1
    if (i >= 1 and i <= 1):
        start_time = time.time()
        text = line[:-1]
        outfile = "/home/mhilmiasyrofi/Documents/cross-asr/tts_espeak/temp_data/" + "audio_" + str(i) + ".wav"
        wavfile = "/home/mhilmiasyrofi/Documents/cross-asr/" + \
            wav_folder + "audio_" + str(i) + ".wav"
        cmd = "espeak \"" + text + "\" --stdout > " + outfile
        # print(cmd)
        os.system(cmd)
        os.system('ffmpeg -i ' + outfile +
                  ' -acodec pcm_s16le -ac 1 -ar 16000 ' + wavfile + ' -y')
        end_time = time.time()
        time_execution = round(end_time - start_time, 2)
        file.write("%d, %.2f\n" % (i, time_execution))
        print("Generated audio_%d.wav with Espeak" % i)

file.close()
