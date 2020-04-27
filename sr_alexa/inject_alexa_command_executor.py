import os
import wave
import struct
import glob
from os import walk



if __name__ == '__main__':

    source_path = "data/"

    output_folder = "sr_alexa/alexa_data/tts_google/generated_speech/"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # for (dirpath, _, filenames) in walk(source_path):
    #     if (len(filenames) > 0) :
    dirpath = "data/tts_google/generated_speech/"
    for i in range(1, 20001):
        filename = "audio_" + str(i) + ".wav"
        fpath = os.path.join(dirpath, filename)
        
        output_file = output_folder + filename

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

        alexa_instruction.close()
        skill_executor.close()
        alexa_instruction.close()

        output = wave.open(output_file, 'wb')
        output.setparams(data[0][0])
        for params, frames in data:
            output.writeframes(frames)
        output.close()

        print("Save audio at %s" % output_file)
