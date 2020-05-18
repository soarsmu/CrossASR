import os
import glob
import subprocess

if __name__ == '__main__':

    TTS = "google"

    data = "data/" + TTS + "/wav2letter/"
    raw_data = "data/" + TTS + "/wav2letter/raw_transcription/"

    wav2letter_tanslation = open(data + "transcription.txt", "w+")

    for i in range(1, 20001):
        filename = "audio_" + str(i) + ".wav.txt"
        fpath = os.path.join(raw_data, filename)
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

        wav2letter_tanslation.write(TTS + ", " + str(i) + ", " + translation + "\n")
                                
        file.close()

    wav2letter_tanslation.close()
