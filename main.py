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
MAX_SEQUENCE_LENGTH = 1000
MAX_CHAR_PER_TOKEN = 5
MAX_NB_WORDS = 20000
CHAR_EMBEDDING_DIM = 100
WORD_EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.25

texts = []  # list of text samples
labels_index = {}  # dictionary mapping label name to numeric id
labels = []  # list of label ids


def load_text(corpus='rcv1', percent=1.0):
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


def load_data(texts, labels):
    # finally, vectorize the text samples into a 2D integer tensor
    tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))
    data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
    new_labels = to_categorical(np.asarray(labels))
    print('Shape of data tensor:', data.shape)
    print('Shape of label tensor:', new_labels.shape)


    # split the data into a training set and a validation set
    indices = np.arange(data.shape[0])
    np.random.shuffle(indices)

    data = data[indices]
    new_labels = new_labels[indices]
    num_validation_samples = int(VALIDATION_SPLIT * data.shape[0])
    x_train = data[:-num_validation_samples]
    y_train = new_labels[:-num_validation_samples]
    x_val = data[-num_validation_samples:]
    y_val = new_labels[-num_validation_samples:]
    gold_labels = np.asarray(labels)[indices][-num_validation_samples:]

    return x_train, y_train, x_val, y_val, word_index, gold_labels



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
                                trainable=True, name='word_embed')
    return embedding_layer


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
    # load texts
    texts, labels = load_text(CORPUS_TYPE)
    print("num of authors:",len(set(labels)))
    import statistics
    MAX_SEQUENCE_LENGTH = statistics.median([len(text) for text in texts])
    MAX_SEQUENCE_LENGTH = min(MAX_SEQUENCE_LENGTH,1000)


    if NETWORK_TYPE == 'cnn' or NETWORK_TYPE == 'lstm' or NETWORK_TYPE == 'cnn_lstm' or NETWORK_TYPE == 'cnn_simple' or NETWORK_TYPE == 'cnn_simple_2' or NETWORK_TYPE == 'cnn_simple_3' :


        x_train, y_train, x_val, y_val, word_index, Y = load_data(texts, labels)


        # first, build index mapping words in the embeddings set to their embedding vector
        embedding_layer = loadWordEmbeddingMatrix(word_index)

        sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
        embedded_sequences = embedding_layer(sequence_input)
        model = models.build_model(NETWORK_TYPE, embedded_sequences, labels_index, sequence_input)


    elif NETWORK_TYPE == 'char_cnn_2':
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



        #word embedding
        word_embedding_layer = loadWordEmbeddingMatrix(word_index)
        word_sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')

        model = models.build_model(NETWORK_TYPE, word_embedding_layer(word_sequence_input), labels_index, word_sequence_input,
                                    embedded_char_sequences = char_embedding_layer(char_sequence_input), char_sequence_input = char_sequence_input)


    elif NETWORK_TYPE == 'char_cnn':
        alphabet = (list(string.ascii_letters) + list(string.digits) +
                    list(string.punctuation) + ['\n'] + [' '])
        vocab_size = len(alphabet)

        x_train_c, y_train_c, x_val_c, y_val_c, word_index = load_char_data(texts, labels, alphabet)
        x_train_w, y_train_w, x_val_w, y_val_w, word_index = load_data(texts, labels)

        #char embedding
        #char embedding
        char_embedding_layer = Embedding(vocab_size,
                                         CHAR_EMBEDDING_DIM,
                                         input_length=MAX_SEQUENCE_LENGTH * MAX_CHAR_PER_TOKEN,
                                         trainable=True, name='char_embed')
        char_sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH * MAX_CHAR_PER_TOKEN,), name='input')

        model = models.build_model(NETWORK_TYPE, None, labels_index, None,
                                    embedded_char_sequences = char_embedding_layer(char_sequence_input), char_sequence_input = char_sequence_input)

    elif NETWORK_TYPE == 'char_lstm_2':
        chars = sorted(list(set(texts)))
        print('total chars:', len(chars))
        char_indices = dict((c, i) for i, c in enumerate(chars))
        indices_char = dict((i, c) for i, c in enumerate(chars))

        maxlen = 512 #MAX_SEQUENCE_LENGTH * MAX_CHAR_PER_TOKEN
        step = 3
        sentences = []
        next_chars = []
        for i in range(0, len(texts) - maxlen, step):
            sentences.append(texts[i: i + maxlen])
            next_chars.append(texts[i + maxlen])
        print('nb sequences:', len(sentences))


        print('Vectorization...')
        X = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
        #y = np.zeros((len(sentences), maxlen, len(labels)), dtype=np.bool)
        y = to_categorical(np.asarray(labels))
        for i, sentence in enumerate(sentences):
            for t, char in enumerate(sentence):
                X[i, t, char_indices[char]] = 1
            #y[i, char_indices[next_chars[i]]] = 1

        x_train_c = y_train_c = X
        x_val_c = y_val_c = y
        model = models.build_lstm_char(labels_index, maxlen, chars)


    elif NETWORK_TYPE == 'char_lstm':
        alphabet = (list(string.ascii_letters) + list(string.digits) +
                    list(string.punctuation) + ['\n'] + [' '])
        vocab_size = len(alphabet)

        x_train_c, y_train_c, x_val_c, y_val_c, word_index = load_char_data(texts, labels, alphabet)

        # char embedding
        char_sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH * MAX_CHAR_PER_TOKEN,), name='input')

        #char_embedding_layer = Embedding(vocab_size,
        #                                 CHAR_EMBEDDING_DIM,
        #                                 input_length=MAX_SEQUENCE_LENGTH * MAX_CHAR_PER_TOKEN,
        #                                 trainable=True, name='char_embed')
        d = Dense(CHAR_EMBEDDING_DIM, name='char_embed')(char_sequence_input)
        char_embedding_layer = TimeDistributed(d)

        model = models.build_model(NETWORK_TYPE, None, labels_index, None,
                                   embedded_char_sequences=char_embedding_layer,
                                   char_sequence_input=char_sequence_input)

        print(model.summary())
    print('Training model.')

    _callbacks = [
        EarlyStopping(monitor='val_loss', patience=2, verbose=0),
        #ModelCheckpoint(kfold_weights_path, monitor='val_loss', save_best_only=True, verbose=0),
    ]

    if NETWORK_TYPE == 'char_cnn_2':
        model.fit([x_train_w, x_train_c], y_train_c,
                  batch_size=128,
                  epochs=1000,
                  validation_data=([x_val_w,x_val_c], y_val_c), callbacks = _callbacks)

    elif NETWORK_TYPE == 'char_cnn' or NETWORK_TYPE == 'char_lstm' or NETWORK_TYPE == 'char_lstm_2':
        model.fit(x_train_c, y_train_c,
                  batch_size=128,
                  epochs=1000,
                  validation_data=(x_val_c, y_val_c), callbacks = _callbacks)
    else:
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
    gold = Y
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

