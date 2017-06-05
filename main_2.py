'''This script loads pre-trained word embeddings (GloVe embeddings)
into a frozen Keras Embedding layer, and uses it to
train a text classification model on the 20 Newsgroup dataset
(classication of newsgroup messages into 20 different categories).

GloVe embedding data can be found at:
http://nlp.stanford.edu/data/glove.6B.zip
(source page: http://nlp.stanford.edu/projects/glove/)

20 Newsgroup data can be found at:
http://www.cs.cmu.edu/afs/cs.cmu.edu/project/theo-20/www/data/news20.html
'''

from __future__ import print_function

import os
import sys
import numpy as np
import random
import models
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import Model,Input
from keras.layers import Embedding, Dense, TimeDistributed
from keras.callbacks import EarlyStopping

import pickle
import theano
theano.config.openmp = True

import string

BASE_DIR = './data'
GLOVE_DIR = BASE_DIR + '/glove/'
TEXT_DATA_DIR1 = BASE_DIR + '/rcv1/all/'
TEXT_DATA_DIR2 = BASE_DIR + '/enron_data/'
MAX_SEQUENCE_LENGTH = 500
MAX_CHAR_PER_TOKEN = 10
MAX_NB_WORDS = 20000
CHAR_EMBEDDING_DIM = 100
WORD_EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.2

texts = []  # list of text samples
labels_index = {}  # dictionary mapping label name to numeric id
labels = []  # list of label ids

def fake_load_text(corpus='rcv1', percent=0.2):
    num_article = 100
    y = []
    for i in range(0,num_article):
        if i < num_article/2:
            y.append(0)
        else:
            y.append(1)
    X = []
    for i in range(0,num_article):
        if i < num_article/2:
            X.append('aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa')
        else:
            X.append('bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb')

    return X, y

def load_text_as_sentence(corpus='rcv1', percent=0.1):
    if corpus == 'rcv1': corpus = TEXT_DATA_DIR1
    else: corpus = TEXT_DATA_DIR2

    for file_name in sorted(os.listdir(corpus)):
        # file name like: william_wallis=721878newsML.xml.txt
        if False == file_name.endswith(".txt"): continue
        if random.uniform(0.0, 1.0) > percent: continue

        author_name = file_name.split('=')[0]
        if author_name not in labels_index:
            label_id = len(labels_index)
            labels_index[author_name] = label_id

        # open file, read each line
        with open(os.path.join(corpus, file_name)) as f:
            lines = f.readlines()
            lines = [x.strip() for x in lines]
        for line in lines:
            texts.append(line)
            label_id = labels_index[author_name]
            labels.append(label_id)

    print('Found %s texts.' % len(texts))
    return texts,labels


def load_text(corpus='rcv1', percent=0.1):
    if corpus == 'rcv1': corpus = TEXT_DATA_DIR1
    else: corpus = TEXT_DATA_DIR2

    for file_name in sorted(os.listdir(corpus)):
        # file name like: william_wallis=721878newsML.xml.txt
        if False == file_name.endswith(".txt"): continue
        if random.uniform(0.0, 1.0) > percent: continue

        author_name = file_name.split('=')[0]

        if author_name not in labels_index:
            label_id = len(labels_index)
            labels_index[author_name] = label_id

        label_id = labels_index[author_name]
        labels.append(label_id)

        # open file, read each line
        with open(os.path.join(corpus, file_name)) as f:
            lines = f.readlines()
            lines = [x.strip() for x in lines]
        text = ''
        for line in lines:
            text += line
        texts.append(text)
    print('Found %s texts.' % len(texts))
    return texts,labels

def encode_data_2(x, maxlen, vocab, vocab_size, check):
    input_data = np.zeros((len(x), maxlen, vocab_size), dtype=np.byte)

    #sent_array_0 = np.zeros((maxlen, vocab_size))
    #for i in range(0,len(sent_array_0)):
    #    sent_array_0[i,0] = 1

    for dix, sent in enumerate(x):
        sent_array = gen_sent_array(sent,maxlen, vocab, vocab_size, check)
        input_data[dix, :, :] = sent_array
    return input_data

def gen_sent_array(sentence, maxlen, vocab, vocab_size, check):
    sent_array = np.zeros((maxlen, vocab_size))
    chars = list(sentence.lower().replace(' ', ''))
    for cix, c in enumerate(chars):
        if cix >= maxlen: break
        if c in vocab:
            sent_array[cix][vocab[c]] = 1
    return sent_array

def fake_encode_data(x, maxlen, vocab, vocab_size, check):
    input_data = np.zeros((len(x), maxlen, vocab_size), dtype=np.byte)

    sent_array_0 = np.zeros((maxlen, vocab_size))
    for i in range(0,len(sent_array_0)):
        sent_array_0[i,0] = 1

    sent_array_1 = np.zeros((maxlen, vocab_size))
    for i in range(0,len(sent_array_1)):
        sent_array_1[i,1] = 1

    for i in range(0,len(x)):
        if i < len(x)/2:
            sent = sent_array_0
        else:
            sent = sent_array_1
        input_data[i, :, :] = sent
    return input_data


def encode_data_old(x, maxlen, vocab, vocab_size, check):
    input_data = np.zeros((len(x), maxlen, vocab_size))
    for dix, sent in enumerate(x):
        counter = 0
        sent_array = np.zeros((maxlen, vocab_size))
        chars = list(sent.lower().replace(' ', ''))
        for c in chars:
            if counter >= maxlen:
                pass
            else:
                char_array = np.zeros(vocab_size, dtype=np.int)
                if c in check:
                    ix = vocab[c]
                    char_array[ix] = 1
                sent_array[counter, :] = char_array
                counter += 1
        input_data[dix, :, :] = sent_array
    return input_data

if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser("NN for NLP")
    parser.add_argument("-network", help="NN type: cnn, lstm, cnn_lstm, cnn_simple, char_cnn, cnn_simple_2, cnn_simple_3 ,char_cnn_2,char_cnn_3,char_lstm,char_lstm_2")
    parser.add_argument("-corpus", help="training corpus: rcv1, enron")

    args = parser.parse_args()

    possible_network = ['cnn', 'lstm', 'cnn_lstm', 'cnn_simple', 'char_cnn', 'cnn_simple_2', 'cnn_simple_3', 'char_cnn_2', 'char_cnn_3','char_lstm','char_lstm_2']
    possible_corpus  = ['rcv1', 'enron']
    if args.network not in possible_network:
        raise ValueError('not supported network type')
    if args.corpus not in possible_corpus:
        raise ValueError('not supported corpus type')

    NETWORK_TYPE = args.network
    CORPUS_TYPE = args.corpus

    # second, prepare text samples and their labels
    print('Processing text dataset')

    #load texts
    texts,labels = load_text_as_sentence()
    #texts, labels = fake_load_text()
    import statistics
    MAX_SEQUENCE_LENGTH = int(statistics.median([len(text) for text in texts]))


    import py_crepe

    # Filters for conv layers
    nb_filter = 64
    # Number of units in the dense layer
    dense_outputs = 64
    # Conv layer kernel size
    filter_kernels = [7, 7, 3, 3, 3, 3]
    # Number of units in the final output layer. Number of classes.
    cat_output = len(set(labels))

    alphabet = (list(string.ascii_lowercase) + list(string.digits) +
                list(string.punctuation) + [' '])

    #X = encode_data(texts,_maxlen,{k: v for v, k in enumerate(alphabet)},len(alphabet),set(alphabet))
    X =  encode_data_old(texts, MAX_SEQUENCE_LENGTH, {k: v for v, k in enumerate(alphabet)}, len(alphabet), set(alphabet))
    _Y = np.asarray(labels)
    Y = to_categorical(_Y)

    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)

    X = X[indices]
    Y = Y[indices]
    _Y = _Y[indices]

    num_validation_samples = int(VALIDATION_SPLIT * len(X))

    x_train = X[:-num_validation_samples]
    y_train = Y[:-num_validation_samples]
    x_val = X[-num_validation_samples:]
    y_val = Y[-num_validation_samples:]
    _Y = _Y[-num_validation_samples:]

    #model = py_crepe.model(filter_kernels, dense_outputs, _maxlen, len(alphabet), nb_filter, cat_output)
    model = models.build_lstm_char(cat_output, MAX_SEQUENCE_LENGTH, alphabet)


    print(model.summary())



    print('Training model.')

    _callbacks = [
        EarlyStopping(monitor='val_loss', patience=2, verbose=0),
        #ModelCheckpoint(kfold_weights_path, monitor='val_loss', save_best_only=True, verbose=0),
    ]

    model.fit(x_train, y_train,
                  batch_size=128,
                  epochs=1000,
                  validation_data=(x_val, y_val), callbacks = _callbacks)

    predictions = model.predict(x_val)
    prediction_classes = []
    for pred in predictions:
        prediction_classes.append(np.argmax(pred))

    # print confusion matix and other metric
    predictions = model.predict(x_val)
    prediction_classes = []
    for pred in predictions:
        prediction_classes.append(np.argmax(pred))

    import sklearn.metrics as metrics
    gold = _Y
    report = metrics.classification_report(gold, prediction_classes)
    print(report)
    cmat = metrics.confusion_matrix(gold, prediction_classes)
    print(cmat)


    # serialize model to JSON
    model_json = model.to_json()
    with open(CORPUS_TYPE +'.'+ NETWORK_TYPE + ".model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights
    pickle.dump(model.get_weights(), open(CORPUS_TYPE + '.'+NETWORK_TYPE + ".weight.pickle", "wb"))

    print("Saved model to disk")

