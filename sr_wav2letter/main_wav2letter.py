import os
import glob
import subprocess

import time
from datetime import datetime

if __name__ == '__main__':

    TTS = "festival"
    SR = "wav2letter"

    combination = "data/" + TTS + "/" + SR + "/" 

    execution = combination + "execution_time/"
    transcription = combination + "transcription/"

    foutput = transcription + "raw/"

    if not os.path.exists(combination):
        os.makedirs(combination)
    if not os.path.exists(execution):
        os.makedirs(execution)
    if not os.path.exists(transcription):
        os.makedirs(transcription)
    if not os.path.exists(foutput):
        os.makedirs(foutput)
    
    file = open(execution + str(datetime.now()) + ".txt", "w+")


    dirpath = "data/" + TTS + "/generated_speech/"
    
    for i in range(1, 20001):
        start_time = time.time()

        filename = "audio_" + str(i) + ".wav"
        fpath = os.path.join(dirpath, filename)
        print("Processing: " + fpath)
        
        cmd = "docker run --rm -v ~/Documents/cross-asr/:/root/host/ -it --ipc=host --name w2l-multi-thread -a stdin -a stdout -a stderr wav2letter/wav2letter:inference-latest " + \
                "sh -c  \"/root/wav2letter/build/inference/inference/examples/multithreaded_streaming_asr_example " + \
                "--input_files_base_path /root/host/sr_wav2letter/model  " + \
                "--input_audio_files /root/host/" + fpath + " " + \
                "--output_files_base_path /root/host/" + foutput + "\""

        proc = subprocess.Popen([cmd],
                                stdout=subprocess.PIPE, shell=True)
        (out, err) = proc.communicate()
        # print("Translation: " + str(out))
        end_time = time.time()
        time_execution = round(end_time - start_time, 2)
        file.write("%d, %.2f\n" % (i, time_execution))

    
    file.close()
