import os, sys, getopt
import utils
import constant

def printHelp() :
    print('generate_audio.py -t <tts name> -o <output audio dir> -l <lower bound> -u <upper bound>')
    print("or")
    print('generate_audio.py --tts <tts name> --output-dir <output audio dir> --lower-bound <lower bound> --upper-bound <upper bound>')

def main(argv):
    tts = ""
    output_dir = ""
    lower_bound = 0
    upper_bound = 20000
    try:
        opts, args = getopt.getopt(argv,"ht:o:l:u:",["tts=",  "output-dir=", "lower-bound=", "upper-bound="])
    except getopt.GetoptError:
        printHelp()
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            printHelp()
            sys.exit()
        elif opt in ("-t", "--tts"):
            tts = arg
        elif opt in ("-o", "--output-dir"):
            output_dir = arg
        elif opt in ("-l", "--lower-bound"):
            lower_bound = int(arg)
        elif opt in ("-u", "--upper-bound"):
            upper_bound = int(arg)
        
    if tts != "" :
        if output_dir != "" :
            generateAudios(tts, output_dir, lower_bound, upper_bound)
        else :
            print("Please specify the output folder location")
    else :
            print("Please specify the used TTS")

def generateAudios(tts, output_dir, lower_bound=None, upper_bound=None) :

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
        text = corpus[i][:-1] # remove new line in the last sentence
        fpath = output_dir + "/audio-%d.wav" % (i + 1)
        if not os.path.exists(fpath) :
            utils.synthesizeSpeech(tts, text, fpath)
            # print(text)

if __name__ == "__main__":
    main(sys.argv[1:])