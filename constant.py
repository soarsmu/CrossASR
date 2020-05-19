FAIL_TEST_CASE = 1
SUCCESS_TEST_CASE = -1

DETERMINED_TEST_CASE = 1
UNDETERMINED_TEST_CASE = 0

NUM_CORES = 4

# word2vec parameter
W2V_EMBEDDING_SIZE = 30
W2V_EPOCH = 20

CORPUS_FPATH = "corpus/europarl-20k.txt"

# # mac base folder
# BASE_FOLDER = "/Users/mhilmiasyrofi/Documents/cross-asr/"

# linux base folder
BASE_FOLDER = "/home/mhilmiasyrofi/Documents/cross-asr/"


GOOGLE = "google"
APPLE = "apple"
FESTIVAL = "festival"
RV = "rv" # ResponsiveVoice
ESPEAK = "espeak"

DEEPSPEECH = "deepspeech"
ALEXA = "alexa"
GCLOUD = "gcloud"
CHROMESPEECH = "gspeech"
WIT = "wit"
WAV2LETTER = "wav2letter"
PADDLEDEEPSPEECH = "paddledeepspeech"

TTS = GOOGLE
SR = [DEEPSPEECH, WIT, WAV2LETTER, PADDLEDEEPSPEECH]

