import os
import glob
import subprocess
from datetime import datetime

if __name__ == '__main__':

    translation_writer = open("output/deepspeech_translation " + str(datetime.now()) + ".txt", "w+")
    
    data = "data/"

    # for (dirpath, _, filenames) in os.walk(data):
    #     if (len(filenames) > 0):
    #         if not os.path.exists(dirpath):
    #             os.makedirs(dirpath)
    #         for i in range(1, len(filenames)+1):
    #             filename = "audio_" + str(i) + ".wav"
    #             if (filename in filenames):
    #                 fpath = os.path.join(dirpath, filename)
    #                 print("Processing: " + fpath)
    #                 cmd = 'deepspeech --model sr_deepspeech/deepspeech-0.6.1-models/output_graph.pbmm --lm sr_deepspeech/deepspeech-0.6.1-models/lm.binary --trie sr_deepspeech/deepspeech-0.6.1-models/trie --audio ' + fpath

    #                 proc = subprocess.Popen([cmd],
    #                                         stdout=subprocess.PIPE, shell=True)
    #                 (out, err) = proc.communicate()

    #                 transcription = out.decode("utf-8")
                    
    #                 translation_writer.write(
    #                     "%s" % (dirpath + ", " + filename[6:-4] + ", " + transcription))
    #                 print("Transcription: " + transcription)


    dirpath = "data/tts_google/generated_speech/"
    for i in range(24793, 28540):
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



    translation_writer.close()





