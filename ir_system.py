# ########## ########## #
# Calvin Massonnet
# M2 MoSIG
# ########## ########## #

import argparse
import math
import numpy as np
import nltk.corpus as corpus
from array import array
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import RegexpTokenizer

IR_DATA_PATH = 'ir_data/'

DICTIONARY_FILENAME_DEFAULT = 'dictionary.npy'
DOCUMENT_ID_FILENAME_DEFAULT = 'document_ids'
FREQUENCY_FILENAME_DEFAULT = 'frequencies'
DICTIONARY_FILENAME = ''
DOCUMENT_ID_FILENAME = ''
FREQUENCY_FILENAME = ''


def load_dictionary(dictionary_filename):
    dictionary = np.load(dictionary_filename).tolist()
    dictionary = [[x[0], int(x[1]), int(x[2])] for x in dictionary]
    return dictionary


def load_inverted_file(document_id_filename, frequency_filename):
    document_id_list = array('L')
    with open(document_id_filename, 'rb') as file:
        document_id_list.frombytes(file.read())

    frequency_list = array('B')
    with open(frequency_filename, 'rb') as file:
        frequency_list.frombytes(file.read())
    
    return document_id_list, frequency_list


def get_query_word_indices(query, dictionary):
    stopword_list = corpus.stopwords.words('english')
    stemmer = SnowballStemmer('english')
    
    # Lowercase words & Remove punctuations
    tokenizer = RegexpTokenizer(r'\w+')
    query_list = tokenizer.tokenize(query)
    # Stem words (e.g., jumps -> jump; giving -> give) - Stemmers can make errors (e.g., does -> doe)...
    query_list = [stemmer.stem(word) for word in query_list]
    # Remove stopwords
    query_list = [word for word in query_list if word not in stopword_list]
    # Remove duplicates
    query_list = list(dict.fromkeys(query_list))
    
    #print(query_list)
    
    query_word_indices = []
    dictionary_word_only_list = [x[0] for x in dictionary]
    
    for word in query_list:
        try:
            word_index = dictionary_word_only_list.index(word)
            query_word_indices.append(word_index)
        except ValueError:
            print("Word '" + word + "' not in dictionary. Skipping...")
    
    #print(query_word_indices)

    return query_word_indices


def compute_weights(query_word_indices, dictionary, document_id_list, frequency_list):
    frequency_array = [[0 for j in range(len(query_word_indices))] for i in range((max(document_id_list) + 1))]
    tf_array = [[0 for j in range(len(query_word_indices))] for i in range((max(document_id_list) + 1))]
    weight_array = [[0 for j in range(len(query_word_indices))] for i in range((max(document_id_list) + 1))]
    
    # Get frequencies
    for i, word_index in enumerate(query_word_indices):
        position = dictionary[word_index][1]
        nb_documents = dictionary[word_index][2]
        for j in range(nb_documents):
            doc_index = document_id_list[position + j]
            frequency = frequency_list[position + j]
            frequency_array[doc_index][i] = frequency
    
    #print(frequency_array)
    
    # Compute TF
    nb_terms_list = [sum(frequency_list) for frequency_list in frequency_array]
    if len(query_word_indices) > 1:
        for i in range(len(frequency_array)):
            for j in range(len(frequency_array[i])):
                if nb_terms_list[i] > 0:
                    # Special case for when only one of the query words appear in the document
                    if frequency_array[i][j] != nb_terms_list[i]:
                        tf_array[i][j] = frequency_array[i][j] / nb_terms_list[i]
    else:
        tf_array = frequency_array
    
    #print(tf_array)
    
    # Compute IDF
    idf_list = []
    nb_weight_array_documents = len(frequency_array)
    for i in range(len(query_word_indices)):
        nb_term_accross_documents = sum([x[i] for x in frequency_array[:]])
        idf = math.log10(nb_weight_array_documents / nb_term_accross_documents)
        idf_list.append(idf)
    
    #print(idf_list)
    
    # Compute TF x IDF (aka: weights)
    for i in range(len(weight_array)):
        for j in range(len(weight_array[i])):
            weight_array[i][j] = tf_array[i][j] * idf_list[j]
    
    #print(weight_array)
    
    return weight_array


def compute_values(weight_array):
    value_list = []
    
    for weight_list in weight_array:
        value = sum(weight_list)
        value_list.append(value)
    
    #print(value_list)
    
    return value_list


def get_relevant_documents(value_list, nb_documents=1):
    relevant_document_list = []
    document_index_list = np.argsort(value_list)[::-1]
    for i in range(min(nb_documents, len(document_index_list))):
        if value_list[document_index_list[i]] != 0:
            relevant_document_list.append(document_index_list[i])
    return relevant_document_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Information Retrieval System')
    parser.add_argument('-d', '--dictionary', type=str, default='', help='Filename of dictionary')
    parser.add_argument('-i', '--document_ids', type=str, default='', help='Filename of inverted file\'s document identifiers')
    parser.add_argument('-f', '--frequencies', type=str, default='', help='Filename of inverted file\'s frequencies')
    parser.add_argument('-D', '--dataset_name', type=str, default='', help='Name of the dataset from which to use its according dictionary and inverted file')
    parser.add_argument('-n', '--nb_documents', type=int, default=5, help='Number of returned relevant documents')
    args = parser.parse_args()
    
    nb_documents = args.nb_documents
    dataset_name = args.dataset_name
    DICTIONARY_FILENAME = IR_DATA_PATH + DICTIONARY_FILENAME_DEFAULT
    DOCUMENT_ID_FILENAME = IR_DATA_PATH + DOCUMENT_ID_FILENAME_DEFAULT
    FREQUENCY_FILENAME = IR_DATA_PATH + FREQUENCY_FILENAME_DEFAULT
    if dataset_name:
        DICTIONARY_FILENAME = IR_DATA_PATH + dataset_name + '_' + DICTIONARY_FILENAME_DEFAULT
        DOCUMENT_ID_FILENAME = IR_DATA_PATH + dataset_name + '_' + DOCUMENT_ID_FILENAME_DEFAULT
        FREQUENCY_FILENAME = IR_DATA_PATH + dataset_name + '_' + FREQUENCY_FILENAME_DEFAULT
    # Allows override of dataset_name
    if args.dictionary:
        DICTIONARY_FILENAME = args.dictionary
    if args.document_ids:
        DOCUMENT_ID_FILENAME = args.document_ids
    if args.frequencies:
        FREQUENCY_FILENAME = args.frequencies

    dictionary = load_dictionary(DICTIONARY_FILENAME)
    document_id_list, frequency_list = load_inverted_file(DOCUMENT_ID_FILENAME, FREQUENCY_FILENAME)
    
    query = input('Query: ')
    
    if query:
        query_word_indices = get_query_word_indices(query, dictionary)
        
        weight_array = compute_weights(query_word_indices, dictionary, document_id_list, frequency_list)
        
        value_list = compute_values(weight_array)
        
        relevant_document_list = get_relevant_documents(value_list, nb_documents=nb_documents)
        print(relevant_document_list)
