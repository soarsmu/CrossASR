import os
import time
from datetime import datetime

wav_folder = "data/tts_rv/generated_speech/"
mp3_folder = "tts_rv/mp3_generated_speech/"

if not os.path.exists(wav_folder):
    os.makedirs(wav_folder)

if not os.path.exists(mp3_folder):
    os.makedirs(mp3_folder)

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
    # if (i >= 1 and i <= 1):
    start_time = time.time()
    text = line[:-1]
    
    base_folder = "/home/mhilmiasyrofi/Documents/cross-asr/"
    outfile = base_folder + mp3_folder + "audio_%d.mp3" % i
    wavfile = base_folder + wav_folder + "audio_%d.wav" % i

    cmd = "rvtts --voice english_us_male --text \"" + text + "\" -o " + outfile
    # print(cmd)
    os.system(cmd)
    os.system('ffmpeg -i ' + outfile +
            ' -acodec pcm_s16le -ac 1 -ar 16000 ' + wavfile + ' -y')

    end_time = time.time()
    time_execution = round(end_time - start_time, 2)
    file.write("%d, %.2f\n" % (i, time_execution))
    print("Generated audio_%d.wav with ResponsiveVoice" % i)

file.close()
