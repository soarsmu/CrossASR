file =  open("sr_paddledeepspeech/manifest.txt", "w+")

TTS = "festival"

for i in range(1, 20001) :
    file.write("{\"audio_filepath\": \"/cross-asr/data/" + TTS + "/generated_speech/audio_" + str(i)+ ".wav\", \"duration\": 0, \"text\": \"text\"}\n")

file.close()
