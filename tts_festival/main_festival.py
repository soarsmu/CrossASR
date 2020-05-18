import os
import time
from datetime import datetime

wav_folder = "data/tts_festival/generated_speech/"

if not os.path.exists(wav_folder):
    os.makedirs(wav_folder)

file = open("corpus/europarl.txt")
lines = file.readlines()
file.close()

exec_time_folder = "tts_festival/execution_time/"
if not os.path.exists(exec_time_folder):
    os.makedirs(exec_time_folder)

file = open(exec_time_folder + str(datetime.now()) + ".txt", "w+")

i = 0
for line in lines:
    i = i + 1
    if (i >= 1 and i <= 20000):
        start_time = time.time()
        wav_name = "audio_%d.wav" % i
        cmd = "festival -b \"(utt.save.wave (SayText \\\"" + \
            line[:-1] + "\\\") \\\"/home/mhilmiasyrofi/Documents/cross-asr/" + \
             wav_folder + "audio_" + str(i) + ".wav\\\" 'riff)\""
        # print(cmd)
        os.system(cmd)
        end_time = time.time()
        time_execution = round(end_time - start_time, 2)
        file.write("%d, %.2f\n" % (i, time_execution))
        print("Generated audio_%d.wav with Festival" % i)

file.close()
