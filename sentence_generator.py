import ssl
import re
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
import string
import nltk


def preprocess(sentence):
	sentence = sentence.lower()
	tokenizer = RegexpTokenizer(r'\w+')
	tokens = tokenizer.tokenize(sentence)
	# filtered_words = [w for w in tokens if not w in stopwords.words('english')]
	# return " ".join(filtered_words)
	return " ".join(tokens)


def has_numbers(inputString):
    return any(char.isdigit() for char in inputString)

def write_corpus_data(corpus, file) :
    for words in corpus:
        if (len(words) >= 5 and  len(words) <= 15):
            sentence = " ".join(words)
            if (not has_numbers(sentence)):
                preprocessed = preprocess(sentence)
                if ("_" not in preprocessed and " s " not in preprocessed and " d " not in preprocessed and "mr" not in preprocessed and " d" != preprocessed[-2:] and len(preprocessed) > 17):
                    file.write(preprocessed + "\n")
        # elif (len(words) > 10 ) :
        #     while(len(words) > 10) :
        #         sub_sentece = words[:10]
        #         sentence = " ".join(sub_sentece)
        #         if (not has_numbers(sentence)):
        #             preprocessed = preprocess(sentence)
        #             if ("_" not in preprocessed and " s " not in preprocessed and " d " not in preprocessed and "mr" not in preprocessed and " d" != preprocessed[-2:] and len(preprocessed) > 13):
        #                 file.write(preprocessed + "\n")
        #         words = words[10:]

if __name__ == '__main__':
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context

    # nltk.download('gutenberg')
    # nltk.download('punkt')
    # nltk.download('stopwords')

    file = open("corpus-sentence.txt", "w+")

    file_ids = nltk.corpus.gutenberg.fileids()

    i = 0
    for file_id in file_ids :
        i += 1
        print("Processing: " + file_id)
        corpus = nltk.Text(nltk.corpus.gutenberg.sents(file_id))
        # if (i == 1) :
        #     for words in corpus :
        #         print("word: " + str(words))
        write_corpus_data(corpus, file)

    file.close()

