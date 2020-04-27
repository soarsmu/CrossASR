import os
import glob
import subprocess

if __name__ == '__main__':

    data = "output/wav2letter/"

    wav2letter_tanslation = open("output/wav2letter_translation.txt", "w+")

    for (dirpath, _, filenames) in os.walk(data):
        for i in range(1, len(filenames)+1):
            filename = "audio_" + str(i) + ".wav.txt"
            if (filename in filenames):
                fpath = os.path.join(dirpath, filename)
                print("Processing: " + fpath)
                
                file = open(fpath)

                lines = file.readlines()

                translation = ""

                j = 0
                for line in lines :
                    j += 1
                    if (j != 1) :
                        part = line.split(",")[-1]
                        if part != "" :
                            translation += part[:-1]
                

                translation = translation[:-1]

                wav2letter_tanslation.write("data/tts_google/generated_speech, " + str(i) + ", " + translation + "\n")
                                        
                file.close()

    wav2letter_tanslation.close()
