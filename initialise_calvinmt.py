# ########## ########## #
# Calvin Massonnet
# M2 MoSIG
# ########## ########## #

import ir_system as irs

import argparse
import numpy as np
import nltk.corpus as corpus
import nltk.tokenize as tokenize
from array import array
from nltk.probability import FreqDist
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import RegexpTokenizer

PERCENTAGE = False

DATASET_PATH = ''
DOCUMENTS_FILENAME = 'cmt.all'

DICTIONARY_FILENAME = 'calvinmt_dictionary'
DOCUMENT_ID_FILENAME = 'calvinmt_document_ids'
FREQUENCY_FILENAME = 'calvinmt_frequencies'


def load_dataset():
    STATE_IDLE = 0
    STATE_I = 1
    STATE_T = 2
    STATE_A = 3
    STATE_B = 4
    STATE_W = 5
    state = STATE_IDLE
    filename_list = []
    filename_id_list = []
    document_words_list = []
    with open(DATASET_PATH + DOCUMENTS_FILENAME, 'r') as file:
        for line in file:
            line = line.replace('\n', ' ')
            if line[0] == '.':
                marker = line[1]
                if marker == 'I':
                    state = STATE_I
                    document_id = int(line.split()[1])
                    filename_id_list.append(document_id)
                elif marker == 'T':
                    state = STATE_T
                    filename_list.append('')
                elif marker == 'A':
                    state = STATE_A
                elif marker == 'B':
                    state = STATE_B
                elif marker == 'W':
                    state = STATE_W
                    document_words_list.append([])
            else:
                current_document = len(filename_id_list) - 1
                if state == STATE_I:
                    pass
                elif state == STATE_T:
                    filename_list[current_document] += line
                elif state == STATE_A:
                    pass
                elif state == STATE_B:
                    pass
                elif state == STATE_W:
                    document_words_list[current_document].extend(line.split())
    return filename_list, filename_id_list, document_words_list


def compute():
    filename_list, filename_id_list, document_words_list = load_dataset()
    stopword_list = corpus.stopwords.words('english')
    stemmer = SnowballStemmer('english')

    # 2D array containing terms, their position in the inverted file and the number of documents they appear in
    # [[term_1, pos, nb], [term_2, pos, nb], ...]
    dictionary = []

    # Two 2D arrays representing an inverted file
    document_id_list = array('L')
    frequency_list = array('B')

    for i, (document_id, filename) in enumerate(zip(filename_id_list, filename_list)):
        word_list = document_words_list[i]
        
        # Lowercase words & Remove punctuations
        tokenizer = RegexpTokenizer(r'\w+')
        word_list = tokenizer.tokenize(' '.join(str(word) for word in word_list))
        # Stem words (e.g., jumps -> jump; giving -> give) - Stemmers can make errors (e.g., does -> doe)...
        word_list = [stemmer.stem(word) for word in word_list]
        # Remove stopwords
        word_list = [word for word in word_list if word not in stopword_list]
        # Count words and remove duplicates (e.g., [[tree, 87], [anna, 305], ...])
        word_count_list = list(map(list, FreqDist(word_list).items()))
        
        dictionary_word_column = [element[0] for element in dictionary]
        for word_count in word_count_list:
            # XXX - frequency constraint
            if word_count[1] > 255:
                word_count[1] = 255
            if not dictionary or word_count[0] not in dictionary_word_column:
                # Append to dictionary
                new_word_element = [word_count[0], len(document_id_list), 1]
                dictionary.append(new_word_element)
                dictionary_word_column.append(new_word_element[0])
                # Append to inverted file
                document_id_list.append(document_id)
                frequency_list.append(word_count[1])
            else:
                # Update dictionary
                word_id = dictionary_word_column.index(word_count[0])
                dictionary[word_id][2] += 1
                # Update inverted file (& dictionary)
                new_position = dictionary[word_id][1] + dictionary[word_id][2] - 1
                document_id_list.insert(new_position, document_id)
                frequency_list.insert(new_position, word_count[1])
                for element_id, element in enumerate(dictionary):
                    if element[1] >= new_position and element_id != word_id:
                        element[1] += 1
        PRINT_PERCENTAGE(document_id + 1, len(filename_list))

    dictionary = np.array(dictionary)
    np.save(irs.IR_DATA_PATH + DICTIONARY_FILENAME, dictionary)

    with open(irs.IR_DATA_PATH + DOCUMENT_ID_FILENAME, 'wb') as file:
        file.write(bytes(document_id_list))

    with open(irs.IR_DATA_PATH + FREQUENCY_FILENAME, 'wb') as file:
        file.write(frequency_list)


def PRINT_PERCENTAGE(value, max):
    if PERCENTAGE:
        percentage = value * 100 / max
        print('[%d%%]\r'%percentage, end='')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--dataset_path', type=str, default='datasets/calvinmt/', help='Path to the dataset')
    parser.add_argument('-P', '--percentage', action='store_true', help='Print percentage of completion')
    args = parser.parse_args()
    
    DATASET_PATH = args.dataset_path
    PERCENTAGE = args.percentage
    
    compute()
