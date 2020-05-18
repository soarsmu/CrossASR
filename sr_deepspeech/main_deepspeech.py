import os
import glob
import subprocess
import time
from datetime import datetime

if __name__ == '__main__':

    TTS = "festival"
    SR = "deepspeech"

    combination = "data/" + TTS + "/" + SR + "/" 

    execution = combination + "execution_time/"
    transcription = combination + "transcription/"

    if not os.path.exists(combination):
        os.makedirs(combination)
    if not os.path.exists(execution):
        os.makedirs(execution)
    if not os.path.exists(transcription):
        os.makedirs(transcription)

    timestamp = str(datetime.now())

    translation_writer = open(transcription + timestamp + ".txt", "w+")

    file = open(execution + timestamp + ".txt", "w+")


    dirpath = "data/" + TTS + "/generated_speech/"
    for i in range(1, 20001):
        start_time = time.time()

        filename = "audio_" + str(i) + ".wav"
        fpath = os.path.join(dirpath, filename)
        print("Processing: " + fpath)
        cmd = 'deepspeech --model sr_deepspeech/deepspeech-0.6.1-models/output_graph.pbmm --lm sr_deepspeech/deepspeech-0.6.1-models/lm.binary --trie sr_deepspeech/deepspeech-0.6.1-models/trie --audio ' + fpath

        proc = subprocess.Popen([cmd],
                                stdout=subprocess.PIPE, shell=True)
        (out, err) = proc.communicate()

        transcription = out.decode("utf-8")

        translation_writer.write(
            "%s" % (dirpath + ", " + filename[6:-4] + ", " + transcription))
        print("Transcription: " + transcription)

        end_time = time.time()
        time_execution = round(end_time - start_time, 2)
        file.write("%d, %.2f\n" % (i, time_execution))


    file.close()

    translation_writer.close()





