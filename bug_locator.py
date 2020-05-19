from jiwer import wer

import constant, utils

TTS = constant.TTS

SR = constant.SR

def get_corpus(fpath) :

    file = open(fpath)

    lines = file.readlines()
    data = {}
    i = 0
    for line in lines:
        i = i + 1
        data[i] = line

    file.close()

    return data

def calculate_transcription_error(sr, data) :
    filename = "data/" + TTS + "/" + sr + "/transcription.txt"
    file = open(filename)
    lines = file.readlines()
    errors = {}
    for line in lines:
        parts = line.split(",")
        if (len(parts) == 3):
            tts = parts[0]
            audio_id = parts[1]
            idx = int(audio_id)
            translation = parts[2]
            translation = utils.preprocess_text(translation)
            error = round(wer(translation, data[int(audio_id)]), 2)
            if (idx not in errors.keys()):
                errors[idx] = error
            else:
                if (errors[idx] > error):
                    errors[idx] = error
    file.close()

    return errors

def localize_bug(errors) :
    undetermined_test_cases = []
    bugs = {}

    for sr in SR :
        bugs[sr] = []
        for idx, error in errors[sr].items():
            if (error != 0 and idx not in undetermined_test_cases) :
                is_other_speech_recognition_can_translate = False
                for s in SR :
                    if s != sr :
                        if (idx in errors[s].keys()) :
                            if (errors[s][idx] == 0) :
                                is_other_speech_recognition_can_translate = True
                                break
                if (is_other_speech_recognition_can_translate) :
                    bugs[sr].append(idx)
                else :
                    undetermined_test_cases.append(idx)
    
    return bugs, undetermined_test_cases

def write_bugs(bugs) :
    for sr in SR:
        filename = "data/" + TTS + "/" + sr + "/bug.txt"
        file = open(filename, "w+")
        for idx_bug in sorted(bugs[sr]):
            file.write(str(idx_bug) + "\n")
        file.close()

if __name__ == '__main__':

    corpus = {}
    corpus = get_corpus(constant.CORPUS_FPATH)

    errors = {}
    for sr in SR :
        errors[sr] = calculate_transcription_error(sr, corpus)

    bugs, undetermined_test_cases = localize_bug(errors)

    undetermined_test_cases_file = open("data/" + TTS + "/undetermined_test_cases.txt", "w+")
    for idx in undetermined_test_cases :
        undetermined_test_cases_file.write(str(idx) + "\n")
    undetermined_test_cases_file.close()
    
    write_bugs(bugs)




    

    




    
