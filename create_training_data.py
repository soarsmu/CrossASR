import constant, utils

TTS = constant.TTS
SR = constant.SR

def get_bug_location(fpath) :
    file = open(fpath)
    lines = file.readlines()
    bugs = []
    for line in lines :
        bugs.append(int(line.split(",")[-1]))
    file.close()
    return bugs

def write_label(bugs, writer) :
    if (i in bugs):
        writer.write(", " + str(constant.FAIL_TEST_CASE) +"\n")
    else :
        writer.write(", " + str(constant.SUCCESS_TEST_CASE) + "\n")

def write_undetermined_test_case(writer) :
    writer.write(", " + str(constant.UNDETERMINED_TEST_CASE) + "\n")

if __name__ == '__main__':

    bugs = {}
    for sr in SR :
        bugs[sr] = get_bug_location("data/" + TTS + "/" + sr + "/bug.txt")

    undetermined_test_case_file = open("data/" + TTS + "/undetermined_test_cases.txt")
    lines = undetermined_test_case_file.readlines()
    undetermined_test_cases = []
    for id in lines :
        undetermined_test_cases.append(int(id))
    undetermined_test_case_file.close()


    file = open(constant.CORPUS_FPATH)
    corpus = file.readlines()
    file.close() 

    training_data = {}
    for sr in SR :
        training_data[sr] = open("data/" + TTS + "/" + sr + "/training_data.txt", "w+")

    i = 0
    for sentence in corpus :
        i += 1

        sentence = utils.preprocess_text(sentence[:-1])
        
        for sr in SR :
            training_data[sr].write(sentence)
        
        if (i not in undetermined_test_cases) :
            for sr in SR :
                write_label(bugs[sr], training_data[sr])
        else :
            for sr in SR :
                write_undetermined_test_case(training_data[sr])


    for sr in SR :
        training_data[sr].close()


