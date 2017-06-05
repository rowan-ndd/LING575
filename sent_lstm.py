
import os
import sys
import numpy as np
import random
import models
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import Model,Input
from keras.layers import Embedding
from keras.callbacks import EarlyStopping
from keras.layers.wrappers import Bidirectional

import pickle
import theano
from keras.layers.recurrent import LSTM
from keras.layers.core import Dropout, Dense
theano.config.openmp = True

import string
import nltk

BASE_DIR = './data'
GLOVE_DIR = BASE_DIR + '/glove/'
TEXT_DATA_DIR1 = BASE_DIR + '/rcv1/all/'
TEXT_DATA_DIR2 = BASE_DIR + '/enron_data/'
MAX_SEQUENCE_LENGTH = 1000
MAX_CHAR_PER_TOKEN = 5
MAX_NB_WORDS = 20000
CHAR_EMBEDDING_DIM = 300
WORD_EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.2
MAX_SENTENCES_PER_ARTICLE=20
MAX_WORDS_PER_SENT=15

texts = []  # list of text samples
labels_index = {}  # dictionary mapping label name to numeric id
labels = []  # list of label ids


def load_text(corpus):
    if corpus == 'rcv1': corpus = TEXT_DATA_DIR1
    else: corpus = TEXT_DATA_DIR2
    
    for file_name in sorted(os.listdir(corpus)):
        # file name like: william_wallis=721878newsML.xml.txt
        if False == file_name.endswith(".txt"): continue
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
    return texts,labels # list of strings


def load_data(labels):
    #to get word indices for tokens in data
    tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts) #default: puncs,tabs,newlines removed. This is where stop words can be filtered out by defining a new list. Output list of word indices
    word_index = tokenizer.word_index #start from 1
    print('Found %s unique tokens.' % len(word_index))
    data  = np.zeros((len(texts), MAX_SENTENCES_PER_ARTICLE, MAX_WORDS_PER_SENT))
    for i, text in enumerate(texts):
        sents = nltk.sent_tokenize(text)
        for j, sent in enumerate(sents): 
            if j<MAX_SENTENCES_PER_ARTICLE:
                for k, word in enumerate(text_to_word_sequence(sent)): 
                    if k<MAX_WORDS_PER_SENT:
                        data[i,j,k]=word_index.get(word)
    
#     data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)#front padding with 0s up to max_seq_len, output (num_texts, max_seq_len)
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





def loadWordEmbeddingMatrix():
    print('Indexing word vectors.')
    embeddings_index = {}
    f = open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt'))
    for line in f:
        values = line.split()
        word = values[0] #str
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    print('Found %s word vectors.' % len(embeddings_index))
    # prepare embedding matrix
    num_words = min(MAX_NB_WORDS, len(word_index)) # get the top MAX_NB_WORDS
    embedding_matrix = np.zeros((num_words, WORD_EMBEDDING_DIM))
    for word, i in word_index.items():
        if i >= MAX_NB_WORDS:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector

    
    return embedding_matrix

def loadSentenceEmbeddings(data):
    embedding_layer = np.zeros((len(data), MAX_SENTENCES_PER_ARTICLE, WORD_EMBEDDING_DIM))
    for i, text in enumerate(data):
        for j, sent in enumerate(text): 
            sent_embedding=np.zeros(WORD_EMBEDDING_DIM) 
            for w in sent:
                if w< MAX_NB_WORDS:
                    sent_embedding+=embedding_matrix[w]
            num_words = np.count_nonzero(sent)
            if num_words>0:                
                embedding_layer[i, j]=sent_embedding/num_words
    return embedding_layer

if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser("NN for NLP")
    parser.add_argument("-network", help="NN type: cnn, lstm, cnn_lstm, cnn_simple, char_cnn, cnn_simple_2, cnn_simple_3 ,char_cnn_2,char_cnn_3")
    parser.add_argument("-corpus", help="training corpus: rcv1, enron")

    args = parser.parse_args()

    possible_network = ['sent','cnn', 'lstm', 'cnn_lstm', 'cnn_simple', 'char_cnn', 'cnn_simple_2', 'cnn_simple_3', 'char_cnn_2', 'char_cnn_3']
    possible_corpus  = ['rcv1', 'enron']
    if args.network not in possible_network:
        raise ValueError('not supported network type')
    if args.corpus not in possible_corpus:
        raise ValueError('not supported corpus type')

    NETWORK_TYPE = args.network
    CORPUS_TYPE = args.corpus

    # second, prepare text samples and their labels
    print('Processing text dataset')

    if NETWORK_TYPE== 'sent':
        #load texts
        texts,labels = load_text(CORPUS_TYPE)
        Y = labels
        x_train, y_train, x_val, y_val, word_index = load_data(labels)
        embedding_matrix = loadWordEmbeddingMatrix()
        x_train = loadSentenceEmbeddings(x_train)
        print(x_train)
        x_val = loadSentenceEmbeddings(x_val)
        print('x_train shape after transf:', x_train.shape)
        print('x_val shape after transf:', x_val.shape)
        input = Input(shape=(MAX_SENTENCES_PER_ARTICLE, WORD_EMBEDDING_DIM,), name='input')
        x = Bidirectional(LSTM(64, dropout=0.2, recurrent_dropout=0.2,))(input)
        x = Dropout(0.2)(x)
        #output dim is TOTAL_AUTHOR_NUM
#         print('num of authors:', y_train)
        pred = Dense(y_train.shape[1], activation='softmax')(x)
        model = Model(input, pred)
        model.compile(loss='categorical_crossentropy',
                      optimizer='rmsprop',
                      metrics=['acc'])

        
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

