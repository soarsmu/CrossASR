import sys, getopt
import utils
import constant

def printHelp() :
    print("For Speech Synthesize")
    print('trial.py -t <tts name> -o <outputfile>')
    print("For Speech Recognition: ")
    print('trial.py -a <asr name> -i <inputfile>')


def main(argv):
    text = "international conference on software engineering, maintenance and evolution"
#     text = "hello world"
    tts = ""
    asr = ""
    inputfile = ""
    outputfile = ""
    try:
        opts, args = getopt.getopt(argv,"ht:a:i:o:s:",["tts=", "asr=", "ifile=", "ofile="])
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
        elif opt in ("-i", "--ifile"):
            inputfile = arg
        elif opt in ("-o", "--ofile"):
            outputfile = arg
        
    if tts != "" :
        if outputfile != "" :
            utils.synthesizeSpeech(tts, text, outputfile)
        else :
            print("Please specify the output file location")

    if asr != "" :
        if inputfile != "" :
            text = utils.recognizeSpeech(asr, inputfile)
            print("Transcription:", text)
        else :
            print("Please specify the audio file location")
    
    
#     print('TTS is', tts)
#     print('ASR is', asr)
#     print('Input file is', inputfile)
#     print('Output file is', outputfile)

if __name__ == "__main__":
    main(sys.argv[1:])