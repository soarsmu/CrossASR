import os
import glob
import subprocess

import time
from datetime import datetime

if __name__ == '__main__':

    data = "data/"

    file = open("sr_wav2letter/execution_time/" +
                str(datetime.now()) + ".txt", "w+")


    # for (dirpath, _, filenames) in os.walk(data):
    #     if (len(filenames) > 0):
    #         if not os.path.exists(dirpath):
    #             os.makedirs(dirpath)
    #         for i in range(1, len(filenames)+1):
    #             filename = "audio_" + str(i) + ".wav"
    #             if (filename in filenames):
    #                 fpath = os.path.join(dirpath, filename)
    #                 print("Processing: " + fpath)
    #                 cmd = "docker run --rm -v ~/Documents/test-case-generation/:/root/host/ -it --ipc=host --name w2l-multi-thread -a stdin -a stdout -a stderr wav2letter/wav2letter:inference-latest sh -c  \"/root/wav2letter/build/inference/inference/examples/multithreaded_streaming_asr_example  --input_files_base_path /root/host/sr_wav2letter/model  --input_audio_files /root/host/" + fpath + "  --output_files_base_path /root/host/output/wav2letter/\""
                    
    #                 proc = subprocess.Popen([cmd],
    #                                         stdout=subprocess.PIPE, shell=True)
    #                 (out, err) = proc.communicate()

    dirpath = "data/tts_google/generated_speech/"
    for i in range(453, 28540):
        start_time = time.time()

        filename = "audio_" + str(i) + ".wav"
        fpath = os.path.join(dirpath, filename)
        print("Processing: " + fpath)
        cmd = "docker run --rm -v ~/Documents/cross-asr/:/root/host/ -it --ipc=host --name w2l-multi-thread -a stdin -a stdout -a stderr wav2letter/wav2letter:inference-latest sh -c  \"/root/wav2letter/build/inference/inference/examples/multithreaded_streaming_asr_example  --input_files_base_path /root/host/sr_wav2letter/model  --input_audio_files /root/host/" + fpath + "  --output_files_base_path /root/host/output/wav2letter/\""

        proc = subprocess.Popen([cmd],
                                stdout=subprocess.PIPE, shell=True)
        (out, err) = proc.communicate()
        # print("Translation: " + str(out))
        end_time = time.time()
        time_execution = round(end_time - start_time, 2)
        file.write("%d, %.2f\n" % (i, time_execution))

    
    file.close()
