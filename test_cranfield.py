# ########## ########## #
# Calvin Massonnet
# M2 MoSIG
# ########## ########## #

import ir_system as irs

import argparse

DATASET_PATH = ''

QUERY_FILENAME = 'cran.qry'
RELEVANCE_FILENAME = 'cranqrel'

DICTIONARY_FILENAME = 'cranfield_dictionary.npy'
DOCUMENT_ID_FILENAME = 'cranfield_document_ids'
FREQUENCY_FILENAME = 'cranfield_frequencies'


def load_queries():
    STATE_IDLE = 0
    STATE_I = 1
    STATE_W = 2
    state = STATE_IDLE
    query_id_list = []
    query_list = []
    with open(DATASET_PATH + QUERY_FILENAME, 'r') as file:
        for line in file:
            line = line.replace('\n', ' ')
            if line[0] == '.':
                marker = line[1]
                if marker == 'I':
                    state = STATE_I
                    document_id = int(line.split()[1])
                    query_id_list.append(document_id)
                elif marker == 'W':
                    state = STATE_W
                    query_list.append('')
            else:
                current_document = len(query_id_list) - 1
                if state == STATE_I:
                    pass
                elif state == STATE_W:
                    query_list[current_document] += line
    return query_id_list, query_list


def load_relevances():
    relevance_list = []
    with open(DATASET_PATH + RELEVANCE_FILENAME, 'r') as file:
        for line in file:
            relevance_list.append([int(x) for x in line.split()])
    return relevance_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test the IR System on the Cranfield dataset')
    parser.add_argument('-p', '--dataset_path', type=str, default='datasets/cranfield/', help='Path to the dataset')
    parser.add_argument('-n', '--nb_documents', type=int, default=5, help='Number of returned relevant documents')
    args = parser.parse_args()
    
    nb_documents = args.nb_documents
    DATASET_PATH = args.dataset_path
    
    query_id_list, query_list = load_queries()
    relevance_list = load_relevances()
    
    relevance_list_per_query = [[] for query_id in range(len(query_id_list))]
    for relevance in relevance_list:
        relevance_list_per_query[relevance[0] - 1].append(relevance[1])
    
    dictionary = irs.load_dictionary(irs.IR_DATA_PATH + DICTIONARY_FILENAME)
    document_id_list, frequency_list = irs.load_inverted_file(irs.IR_DATA_PATH + DOCUMENT_ID_FILENAME, irs.IR_DATA_PATH + FREQUENCY_FILENAME)
    
    for i, (query_id, query) in enumerate(zip(query_id_list, query_list)):
        query_word_indices = irs.get_query_word_indices(query, dictionary)
        weight_array = irs.compute_weights(query_word_indices, dictionary, document_id_list, frequency_list)
        value_list = irs.compute_values(weight_array)
        relevant_document_list = irs.get_relevant_documents(value_list, nb_documents=nb_documents)
        
        nb_similar_relevance = 0
        for relevant_document in relevant_document_list:
            if relevant_document in relevance_list_per_query[i]:
                nb_similar_relevance += 1
        print(str(query_id) + ' (' + str(i + 1) + '): ' + str(nb_similar_relevance) + '/' + str(len(relevance_list_per_query[i])) + ' (' + str(len(relevant_document_list)) + ')')
