

def preprocess_translation(translation):
    translation = translation[:-1].lower()
    words = translation.split(" ")
    preprocessed = []
    for w in words:
        substitution = ""
        if w == "mister":
            substitution = "mr"
        elif w == "mr.":
            substitution = "mr"
        elif w == "i'm":
            substitution = "i am"
        elif w == "1":
            substitution = "one"
        elif w == "2":
            substitution = "two"
        elif w == "3":
            substitution = "three"
        elif w == "4":
            substitution = "four"
        elif w == "5":
            substitution = "five"
        elif w == "6":
            substitution = "six"
        elif w == "7":
            substitution = "seven"
        elif w == "8":
            substitution = "eight"
        elif w == "9":
            substitution = "nine"
        else:
            substitution = w
        preprocessed.append(substitution)
    return " ".join(preprocessed)[1:] + "\n"

def get_bug_location(fpath) :
    file = open(fpath)
    lines = file.readlines()
    bugs = []
    for line in lines :
        bugs.append(int(line.split(",")[-1]))
    file.close()
    return bugs

def get_translation(fpath) :
    file = open(fpath)
    lines = file.readlines()
    translations = {}
    for line in lines:
        parts = line.split(",")
        if (line != "\n" and len(parts) == 3) : 
            audio_id = parts[1]
            idx = int(audio_id)
            translation = parts[2]
            translation = preprocess_translation(translation)
            translations[idx] = translation
    return translations
 


if __name__ == '__main__':

    alexa_translations = get_translation("output/alexa_translation.txt")
    deepspeech_translations = get_translation("output/deepspeech_translation.txt")
    gcloud_translations = get_translation("output/gcloud_translation.txt")
    gspeech_translations = get_translation("output/gspeech_translation.txt")
    wav2letter_translations = get_translation("output/wav2letter_translation.txt")
    wit_translations = get_translation("output/wit_translation.txt")

    alexa_bugs = get_bug_location("bug/sr/alexa_bug.txt")
    deepspeech_bugs = get_bug_location("bug/sr/deepspeech_bug.txt")
    gcloud_bugs = get_bug_location("bug/sr/gcloud_bug.txt")
    gspeech_bugs = get_bug_location("bug/sr/gspeech_bug.txt")
    wav2letter_bugs = get_bug_location("bug/sr/wav2letter_bug.txt")
    wit_bugs = get_bug_location("bug/sr/wit_bug.txt")
    all_failed_tests = open("bug/sr/all_failed_tests.txt")
    lines = all_failed_tests.readlines()
    failed_tests = []
    for id in lines :
        failed_tests.append(int(id))
    all_failed_tests.close()


    file = open("corpus-sentence.txt")
    corpus = file.readlines()
    file.close() 

    alexa_verify_data = open("verify/alexa.txt", "w+")
    deepspeech_verify_data = open("verify/deepspeech.txt", "w+")
    gcloud_verify_data = open("verify/gcloud.txt", "w+")
    gspeech_verify_data = open("verify/gspeech.txt", "w+")
    wav2letter_verify_data = open("verify/wav2letter.txt", "w+")
    wit_verify_data = open("verify/wit.txt", "w+")


    i = 0
    for sentence in corpus :
        i += 1
        if (i not in failed_tests) :
            
            if (i in alexa_bugs) :
                alexa_verify_data.write(str(i) + ", " + alexa_translations[i][:-1] + ", " + sentence)
            
            if (i in deepspeech_bugs) :
                deepspeech_verify_data.write(str(i) + ", " + deepspeech_translations[i][:-1] + ", " + sentence)
            
            if (i in gcloud_bugs) :
                gcloud_verify_data.write(str(i) + ", " + gcloud_translations[i][:-1] + ", " + sentence)
            
            if (i in gspeech_bugs) :
                gspeech_verify_data.write(str(i) + ", " + gspeech_translations[i][:-1] + ", " + sentence)
            
            if (i in wav2letter_bugs) :
                wav2letter_verify_data.write(str(i) + ", " + wav2letter_translations[i][:-1] + ", " + sentence)
            
            if (i in wit_bugs) :
                wit_verify_data.write(str(i) + ", " + wit_translations[i][:-1] + ", " + sentence)
            
            
    alexa_verify_data.close()
    deepspeech_verify_data.close()
    gcloud_verify_data.close()
    gspeech_verify_data.close()
    wav2letter_verify_data.close()
    wit_verify_data.close()
    

