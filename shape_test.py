from keras.layers import Dense, Flatten, Dropout, Bidirectional, Input, merge
from keras.layers import Conv1D, MaxPooling1D, LSTM, GlobalMaxPooling1D, TimeDistributed
from keras.models import Model
from keras.models import Sequential
from keras.callbacks import EarlyStopping
import random
import numpy as np
import keras
import sklearn.metrics as metrics

def fake_data(word_embed_dim, sentence_per_article, num_article):

    x = []
    for i in range(0,num_article):
        _x = np.random.rand(word_embed_dim,sentence_per_article)
        x.append(_x)

    y = np.random.rand(num_article)
    y = [ int(num_article*element) for element in y]

    return x,y

#input dim 100
WORD_EMBED_DIM = 100
MAX_SENTENCES_PER_ARTICLE = 10
TRAIN_SAMPLE_SIZE = 128
TEST_SAMPLE_SIZE = 29


#labels: 5 authors: 0,1,2,3,4
TOTAL_AUTHOR_NUM = 2

import numpy as np

mat0 = np.full((WORD_EMBED_DIM, MAX_SENTENCES_PER_ARTICLE), 0)
mat1 = np.full((WORD_EMBED_DIM, MAX_SENTENCES_PER_ARTICLE), 1)

x_train = np.zeros((TRAIN_SAMPLE_SIZE, WORD_EMBED_DIM, MAX_SENTENCES_PER_ARTICLE))
for i,e in enumerate(x_train):
    if i <= len(x_train)/TOTAL_AUTHOR_NUM:
        x_train[i] = mat0
    else:
        x_train[i] = mat1

x_test = np.zeros((TEST_SAMPLE_SIZE, WORD_EMBED_DIM, MAX_SENTENCES_PER_ARTICLE))
for i,e in enumerate(x_test):
    if i <= len(x_test)/TOTAL_AUTHOR_NUM:
        x_test[i] = mat0
    else:
        x_test[i] = mat1


_y = np.zeros((TRAIN_SAMPLE_SIZE, 1))
for i,e in enumerate(_y):
    if i <= len(_y)/TOTAL_AUTHOR_NUM:
        _y[i] = 0
    else:
        _y[i] = 1
y_train = keras.utils.to_categorical(_y)

y = np.zeros((TEST_SAMPLE_SIZE, 1))
for i,e in enumerate(y):
    if i <= len(y)/TOTAL_AUTHOR_NUM:
        y[i] = 0
    else:
        y[i] = 1
y_test = keras.utils.to_categorical(y, num_classes=TOTAL_AUTHOR_NUM)



print(x_train.shape, ' ' ,x_test.shape)
print(y_train.shape, ' ' ,y_test.shape)

#shape of input is WORD_EMBED_DIM(averaged by words), MAX_SENTENCES_PER_ARTICLE
input = Input(shape=(WORD_EMBED_DIM, MAX_SENTENCES_PER_ARTICLE,), name='input')
#only use LSTM's last output
x = LSTM(128, return_sequences=False, dropout=0.2, recurrent_dropout=0.2,)(input)
x = Dropout(0.2)(x)
#output dim is TOTAL_AUTHOR_NUM
pred = Dense(TOTAL_AUTHOR_NUM, activation='softmax')(x)
model = Model(input, pred)
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['acc'])

_callbacks = [
    EarlyStopping(monitor='val_loss', patience=2, verbose=0),
    # ModelCheckpoint(kfold_weights_path, monitor='val_loss', save_best_only=True, verbose=0),
]

model.fit(x_train, y_train,
          batch_size=128,
          epochs=10,
          validation_data=(x_test, y_test), callbacks=_callbacks)

#print model dim summary
print(model.summary())

predictions = model.predict(x_test)
prediction_classes = []
for pred in predictions:
    prediction_classes.append(np.argmax(pred))

#print confusion matix and other metric
gold = y
report = metrics.classification_report(gold, prediction_classes)
print(report)
cmat = metrics.confusion_matrix(gold, prediction_classes)
print(cmat)