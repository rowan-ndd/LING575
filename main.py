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
from keras.layers import Embedding
from keras.callbacks import EarlyStopping

import pickle
import theano
theano.config.openmp = True

import string
import data_helpers

BASE_DIR = './data'
GLOVE_DIR = BASE_DIR + '/glove/'
TEXT_DATA_DIR = BASE_DIR + '/rcv1/all/'
MAX_SEQUENCE_LENGTH = 1000
MAX_CHAR_PER_TOKEN = 5
MAX_NB_WORDS = 20000
CHAR_EMBEDDING_DIM = 50
WORD_EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.2

texts = []  # list of text samples
labels_index = {}  # dictionary mapping label name to numeric id
labels = []  # list of label ids


def load_text(percent=0.1):

    for file_name in sorted(os.listdir(TEXT_DATA_DIR)):
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
        with open(os.path.join(TEXT_DATA_DIR, file_name)) as f:
            lines = f.readlines()
            lines = [x.strip() for x in lines]
        text = ''
        for line in lines:
            text += line
        texts.append(text)
    print('Found %s texts.' % len(texts))
    return texts,labels


def load_data(texts, labels):
    # finally, vectorize the text samples into a 2D integer tensor
    tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))
    data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
    labels = to_categorical(np.asarray(labels))
    print('Shape of data tensor:', data.shape)
    print('Shape of label tensor:', labels.shape)
    # split the data into a training set and a validation set
    indices = np.arange(data.shape[0])
    np.random.shuffle(indices)
    data = data[indices]
    labels = labels[indices]
    num_validation_samples = int(VALIDATION_SPLIT * data.shape[0])
    x_train = data[:-num_validation_samples]
    y_train = labels[:-num_validation_samples]
    x_val = data[-num_validation_samples:]
    y_val = labels[-num_validation_samples:]
    return x_train, y_train, x_val, y_val, word_index

def load_char_data(texts, labels, alphabet):

    #convert texts into character sequence
    check = set(alphabet)
    char_seqences = []
    for text in texts:
        chars = list(text.replace(' ', ''))
        new_chars = []
        for c in chars:
            if c in check:
                new_chars.append(c)
        char_seqences.append(new_chars)

    tokenizer = Tokenizer(num_words=MAX_NB_WORDS,nb_words=64, lower=True,char_level=True)
    tokenizer.fit_on_texts(char_seqences)
    sequences = tokenizer.texts_to_sequences(char_seqences)
    word_index = tokenizer.word_index

    print('Found %s unique tokens.' % len(word_index))
    data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH*MAX_CHAR_PER_TOKEN)
    labels = to_categorical(np.asarray(labels))
    print('Shape of data tensor:', data.shape)
    print('Shape of label tensor:', labels.shape)

    # split the data into a training set and a validation set
    indices = np.arange(data.shape[0])
    np.random.shuffle(indices)
    data = data[indices]
    labels = labels[indices]
    num_validation_samples = int(VALIDATION_SPLIT * data.shape[0])
    x_train = data[:-num_validation_samples]
    y_train = labels[:-num_validation_samples]
    x_val = data[-num_validation_samples:]
    y_val = labels[-num_validation_samples:]
    return x_train, y_train, x_val, y_val, word_index

def loadCharEmbeddingMatrix():
    print('Indexing char vectors.')
    embeddings_index = {}
    f = open(os.path.join(GLOVE_DIR, 'glove.840B.300d-char.txt'))
    for line in f:
        values = line.split()
        char = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[char] = coefs
    f.close()
    print('Found %s word vectors.' % len(embeddings_index))
    # prepare embedding matrix
    num_words = min(MAX_NB_WORDS, len(word_index))
    embedding_matrix = np.zeros((num_words, WORD_EMBEDDING_DIM))
    for char, i in word_index.items():
        if i >= MAX_NB_WORDS:
            continue
        embedding_vector = embeddings_index.get(char)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector

    # load pre-trained word embeddings into an Embedding layer
    # note that we set trainable = False so as to keep the embeddings fixed
    embedding_layer = Embedding(num_words,
                                CHAR_EMBEDDING_DIM,
                                weights=[embedding_matrix],
                                input_length=MAX_SEQUENCE_LENGTH*MAX_CHAR_PER_TOKEN,
                                trainable=False, name='word_embed')
    return embedding_layer


def loadWordEmbeddingMatrix(word_index):
    print('Indexing word vectors.')
    embeddings_index = {}
    f = open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt'))
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    print('Found %s word vectors.' % len(embeddings_index))
    # prepare embedding matrix
    num_words = min(MAX_NB_WORDS, len(word_index))
    embedding_matrix = np.zeros((num_words, WORD_EMBEDDING_DIM))
    for word, i in word_index.items():
        if i >= num_words:
            break
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector

    # load pre-trained word embeddings into an Embedding layer
    # note that we set trainable = False so as to keep the embeddings fixed
    embedding_layer = Embedding(num_words,
                                WORD_EMBEDDING_DIM,
                                weights=[embedding_matrix],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=False, name='word_embed')
    return embedding_layer


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser("NN for NLP")
    parser.add_argument("-network", help="NN type: cnn, lstm, cnn_lstm, cnn_simple, char_cnn, cnn_simple_2, cnn_simple_3 ,char_cnn_2,char_cnn_3")
    parser.add_argument("-corpus", help="training corpus: rcv1, enron")

    args = parser.parse_args()

    possible_network = ['cnn', 'lstm', 'cnn_lstm', 'cnn_simple', 'char_cnn', 'cnn_simple_2', 'cnn_simple_3', 'char_cnn_2', 'char_cnn_3']
    possible_corpus  = ['rcv1', 'enron']
    if args.network not in possible_network:
        raise ValueError('not supported network type')
    if args.corpus not in possible_corpus:
        raise ValueError('not supported corpus type')

    NETWORK_TYPE = args.network
    CORPUS_TYPE = args.corpus

    # second, prepare text samples and their labels
    print('Processing text dataset')

    if NETWORK_TYPE == 'cnn' or NETWORK_TYPE == 'lstm' or NETWORK_TYPE == 'cnn_lstm' or NETWORK_TYPE == 'cnn_simple' or NETWORK_TYPE == 'cnn_simple_2' or NETWORK_TYPE == 'cnn_simple_3' :
        #load texts
        texts,labels = load_text()

        x_train, y_train, x_val, y_val, word_index = load_data(texts, labels)

        # first, build index mapping words in the embeddings set to their embedding vector
        embedding_layer = loadWordEmbeddingMatrix(word_index)

        sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
        embedded_sequences = embedding_layer(sequence_input)
        model = models.build_model(NETWORK_TYPE, embedded_sequences, labels_index, sequence_input)


    elif NETWORK_TYPE == 'char_cnn_2':
        #load texts
        texts,labels = load_text()

        alphabet = (list(string.ascii_letters) + list(string.digits) +
                    list(string.punctuation) + ['\n'] + [' '])
        vocab_size = len(alphabet)

        x_train_c, y_train_c, x_val_c, y_val_c, word_index = load_char_data(texts, labels, alphabet)
        x_train_w, y_train_w, x_val_w, y_val_w, word_index = load_data(texts, labels)

        #char embedding
        char_embedding_layer = Embedding(vocab_size,
                                         CHAR_EMBEDDING_DIM,
                                         input_length=MAX_SEQUENCE_LENGTH * MAX_CHAR_PER_TOKEN,
                                         trainable=True, name='char_embed')
        char_sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH * MAX_CHAR_PER_TOKEN,), name='input', dtype='int32')

        #inputs = Input(shape=(maxlen, vocab_size), name='input', dtype='float32')

        #word embedding
        word_embedding_layer = loadWordEmbeddingMatrix(word_index)
        word_sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')

        model = models.build_model(NETWORK_TYPE, word_embedding_layer(word_sequence_input), labels_index, word_sequence_input,
                                    embedded_char_sequences = char_embedding_layer(char_sequence_input), char_sequence_input = char_sequence_input)

    elif NETWORK_TYPE == 'char_cnn':
        #load texts
        texts,labels = load_text()

        alphabet = (list(string.ascii_letters) + list(string.digits) +
                    list(string.punctuation) + ['\n'] + [' '])
        vocab_size = len(alphabet)

        x_train_c, y_train_c, x_val_c, y_val_c, word_index = load_char_data(texts, labels, alphabet)
        x_train_w, y_train_w, x_val_w, y_val_w, word_index = load_data(texts, labels)

        #char embedding
        char_sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH * MAX_CHAR_PER_TOKEN,), name='input', dtype='int32')


        #word embedding
        word_embedding_layer = loadWordEmbeddingMatrix(word_index)
        word_sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
        model = models.build_model(NETWORK_TYPE, word_embedding_layer(word_sequence_input), labels_index, word_sequence_input,
                                    embedded_char_sequences = char_sequence_input, char_sequence_input = char_sequence_input)



    _callbacks = [
        EarlyStopping(monitor='val_loss', patience=2, verbose=0),
        #ModelCheckpoint(kfold_weights_path, monitor='val_loss', save_best_only=True, verbose=0),
    ]
    print(model.summary())

    if NETWORK_TYPE == 'char_cnn_2' :
        model.fit([x_train_w, x_train_c], y_train_c,
                  batch_size=128,
                  epochs=1000,
                  validation_data=([x_val_w,x_val_c], y_val_c), callbacks = _callbacks)
    elif NETWORK_TYPE == 'char_cnn' :
        model.fit(x_train_c, y_train_c,
                  batch_size=128,
                  epochs=1000,
                  validation_data=(x_val_c, y_val_c), callbacks = _callbacks)
    else:
        model.fit(x_train, y_train,
                  batch_size=128,
                  epochs=1000,
                  validation_data=(x_val, y_val), callbacks = _callbacks)


    print('Training model.')


    # serialize model to JSON
    model_json = model.to_json()
    with open(CORPUS_TYPE + NETWORK_TYPE + ".model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights
    pickle.dump(model.get_weights(), open(CORPUS_TYPE + NETWORK_TYPE + ".weight.pickle", "wb"))

    print("Saved model to disk")

