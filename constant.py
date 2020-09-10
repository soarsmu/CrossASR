FAIL_TEST_CASE = 1
SUCCESS_TEST_CASE = -1

DETERMINED_TEST_CASE = 1
UNDETERMINED_TEST_CASE = 0

EUROPARL = "europarl"

DATASET = EUROPARL
CORPUS_FPATH = "corpus/" + DATASET + "-20k.txt"

INITIAL_SEED = 12345

# linux base folder
BASE_FOLDER = "/home/mhilmiasyrofi/Documents/cross-asr/"


GOOGLE = "google"
FESTIVAL = "festival"
RV = "rv" # ResponsiveVoice
ESPEAK = "espeak"

DEEPSPEECH = "deepspeech"
PADDLEDEEPSPEECH = "paddledeepspeech"
WIT = "wit"
WAV2LETTER = "wav2letter"


TTS = [GOOGLE, RV, FESTIVAL, ESPEAK]
ASR = [DEEPSPEECH, PADDLEDEEPSPEECH, WAV2LETTER, WIT]

