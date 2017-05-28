
from keras.layers import Dense, Flatten, Dropout, Bidirectional, Input
from keras.layers import Conv1D, MaxPooling1D, LSTM, GlobalMaxPooling1D
from keras.models import Model

# Convolution Parameters
kernel_size = 5
filters = 64
pool_size = 4
filter_kernels = [7, 3,]

def build_model(type, embedded_sequences, labels_index, sequence_input):

    if type == 'cnn':
        x = Dropout(0.2)(embedded_sequences)
        x = Conv1D(128, 5, activation='relu')(x)
        x = MaxPooling1D(5)(x)
        x = Dropout(0.2)(x)
        x = Conv1D(128, 5, activation='relu')(x)
        x = MaxPooling1D(5)(x)
        x = Dropout(0.2)(x)
        x = Conv1D(128, 5, activation='relu')(x)
        x = MaxPooling1D(35)(x)
        x = Dropout(0.2)(x)
        x = Flatten()(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.2)(x)
        preds = Dense(len(labels_index), activation='softmax')(x)
        model = Model(sequence_input, preds)
        model.compile(loss='categorical_crossentropy',
                       optimizer='adam',
                      metrics=['acc'])

    if type == 'cnn_simple':
        x = Dropout(0.2)(embedded_sequences)
        x = Conv1D(filters, kernel_size, padding='valid', activation='relu', strides=1)(x)
        x = GlobalMaxPooling1D()(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.2)(x)
        preds = Dense(len(labels_index), activation='softmax')(x)
        model = Model(sequence_input, preds)
        model.compile(loss='categorical_crossentropy',
                       optimizer='adam',
                      metrics=['acc'])

    if type == 'cnn_simple_2':
        x = Dropout(0.2)(embedded_sequences)
        x = Conv1D(filters, kernel_size, padding='valid', activation='relu', strides=1)(x)
        x = MaxPooling1D(5)(x)
        x = Dropout(0.2)(x)
        x = Conv1D(int(filters/2), int(kernel_size/2), padding='valid', activation='relu', strides=1)(x)
        x = GlobalMaxPooling1D()(x)
        x = Dropout(0.2)(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.2)(x)
        preds = Dense(len(labels_index), activation='softmax')(x)
        model = Model(sequence_input, preds)
        model.compile(loss='categorical_crossentropy',
                       optimizer='adam',
                      metrics=['acc'])

    if type == 'lstm':
        x = Dropout(0.2)(embedded_sequences)
        x = Bidirectional(LSTM(64, dropout=0.2, recurrent_dropout=0.2))(x)
        preds = Dense(len(labels_index), activation='softmax')(x)

        model = Model(sequence_input, preds)
        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['acc'])

    if type == 'cnn_lstm':
        x = Dropout(0.2)(embedded_sequences)
        x = Conv1D(filters, kernel_size, padding='valid', activation='relu', strides=1)(x)
        x = MaxPooling1D(pool_size=pool_size)(x)
        x = Bidirectional(LSTM(64, dropout=0.2, recurrent_dropout=0.2))(x)
        preds = Dense(len(labels_index), activation='softmax')(x)

        model = Model(sequence_input, preds)
        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['acc'])

    if type == 'char_cnn':
        x = Dropout(0.2)(embedded_sequences)
        x = Conv1D(filters, kernel_size, padding='valid', activation='relu', strides=1)(x)
        x = GlobalMaxPooling1D()(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.2)(x)
        preds = Dense(len(labels_index), activation='softmax')(x)
        model = Model(sequence_input, preds)
        model.compile(loss='categorical_crossentropy',
                       optimizer='adam',
                      metrics=['acc'])

    return model