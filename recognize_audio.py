import os, sys, getopt
import utils
import constant

def printHelp() :
    print('recognize_audio.py -a <asr name> -i <input audio dir>  -o <output transcription dir> -l <lower bound> -u <upper bound>')
    print("or")
    print('recognize_audio.py --asr <asr name> --input-dir <input audio dir> --output-dir <output transcription dir> --lower-bound <lower bound> --upper-bound <upper bound>')

def main(argv):
    tts = ""
    asr = ""
    input_dir = ""
    output_dir = ""
    lower_bound = 0
    upper_bound = 20000
    try:
        opts, args = getopt.getopt(argv,"ht:a:i:o:l:u:",["tts=", "asr=", "input-dir=",  "output-dir=", "lower-bound=", "upper-bound="])
    except getopt.GetoptError:
        printHelp()
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            printHelp()
            sys.exit()
        elif opt in ("-t", "--tts"):
            tts = arg
        elif opt in ("-a", "--asr"):
            asr = arg
        elif opt in ("-i", "--input-dir"):
            input_dir = arg
        elif opt in ("-o", "--output-dir"):
            output_dir = arg
        elif opt in ("-l", "--lower-bound"):
            lower_bound = int(arg)
        elif opt in ("-u", "--upper-bound"):
            upper_bound = int(arg)
    
    if tts == "" :
        print("Please specify the used TTS")
        sys.exit()
    elif tts not in constant.TTS :
        print("Please use the correct TTS name")
        sys.exit()
        
    if asr == "" :
        print("Please specify the used ASR")
        sys.exit()
    elif asr not in constant.ASR :
        print("Please use the correct ASR name")
        sys.exit()

    if input_dir == "" :
        print("Please specify the input audio folder location")
    elif input_dir[-1] != "/" :
        print("Please put backslash (/) in the end of the input audio folder")
    else :
        if output_dir == "" :
            print("Please specify the output folder location for saving transcriptions")
        elif output_dir[-1] != "/" :
            print("Please put backslash (/) in the end of the output transcription folder")    
        else :
            recognizeAudios(tts, asr, input_dir, output_dir, lower_bound, upper_bound)
        

def recognizeAudios(tts, asr, input_dir, output_dir, lower_bound=None, upper_bound=None) :

    input_dir = input_dir + tts + "/"
    output_dir = output_dir + tts + "/" + asr + "/"
    
    if not os.path.exists(output_dir) :
        os.makedirs(output_dir)
    
    file = open(constant.CORPUS_FPATH)
    corpus = file.readlines()
    file.close()

    if lower_bound < 0 :
        print("Lower bound is less than zero")
        sys.exit()
    
    if lower_bound > len(corpus) :
        print("Lower bound is greater than the size of corpus")
        sys.exit()

    if upper_bound > len(corpus) :
        print("Upper bound is greater than the size of corpus")
        sys.exit()
    
    for i in range(lower_bound, upper_bound) :
        audio_path = input_dir + "audio-%d.wav" % (i + 1)
        transcription_path = output_dir + "transcription-%d.txt" % (i + 1)
        if not os.path.exists(transcription_path) or utils.isEmptyFile(transcription_path) :
            if os.path.exists(audio_path) :
                print("Processing audio file: %d" % (i + 1))
                transcription = utils.recognizeSpeech(asr, audio_path)
                file = open(transcription_path, "w+")
                file.write("%s\n" % transcription)
                file.close()
                # print(text)

if __name__ == "__main__":
    main(sys.argv[1:])