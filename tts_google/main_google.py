# English
# 'en-us': 'English (US)'
# 'en-ca': 'English (Canada)'
# 'en-uk': 'English (UK)'
# 'en-gb': 'English (UK)'
# 'en-au': 'English (Australia)'
# 'en-gh': 'English (Ghana)'
# 'en-in': 'English (India)'
# 'en-ie': 'English (Ireland)'
# 'en-nz': 'English (New Zealand)'
# 'en-ng': 'English (Nigeria)'
# 'en-ph': 'English (Philippines)'
# 'en-za': 'English (South Africa)'
# 'en-tz': 'English (Tanzania)'
import os
from gtts import gTTS

mp3_folder = "tts_google/mp3_generated_speech/"
wav_folder = "data/tts_google/generated_speech/"

if not os.path.exists(mp3_folder):
    os.makedirs(mp3_folder)

if not os.path.exists(wav_folder):
    os.makedirs(wav_folder)

file = open("corpus-sentence.txt")
lines = file.readlines()

i = 0
for line in lines:
    i = i + 1
    if (i >= 26599 and i <= 28539):
        tts = gTTS(line, lang='en-us')
        outfile = mp3_folder + "audio_%d.mp3" % i
        wavfile = wav_folder + "audio_%d.wav" % i
        tts.save(outfile)
        os.system('ffmpeg -i /home/mhilmiasyrofi/Documents/test-case-generation/' + outfile +
                ' -acodec pcm_s16le -ac 1 -ar 16000 /home/mhilmiasyrofi/Documents/test-case-generation/' + wavfile + ' -y')
        print("Generated audio_%d.wav" % i)
