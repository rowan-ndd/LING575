
from keras.layers import Dense, Flatten, Dropout, Bidirectional, Input, merge
from keras.layers import Conv1D, MaxPooling1D, LSTM, GlobalMaxPooling1D, TimeDistributed
from keras.models import Model

# Convolution Parameters
kernel_size = 5
filters = 64
pool_size = 4
filter_kernels = [5, 7,]

def build_model(type, word_embedded_sequences, labels_index, word_sequence_input, embedded_char_sequences = None, char_sequence_input = None):

    if type == 'cnn':
        x = Dropout(0.2)(word_embedded_sequences)
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
        model = Model(word_sequence_input, preds)
        model.compile(loss='categorical_crossentropy',
                       optimizer='adam',
                      metrics=['acc'])

    if type == 'cnn_simple':
        x = Dropout(0.2)(word_embedded_sequences)
        x = Conv1D(filters, kernel_size, padding='valid', activation='relu', strides=1)(x)
        x = GlobalMaxPooling1D()(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.2)(x)
        preds = Dense(len(labels_index), activation='softmax')(x)
        model = Model(word_sequence_input, preds)
        model.compile(loss='categorical_crossentropy',
                       optimizer='adam',
                      metrics=['acc'])

    if type == 'cnn_simple_2':
        x = Dropout(0.2)(word_embedded_sequences)
        x = Conv1D(filters, kernel_size, padding='valid', activation='relu', strides=1)(x)
        x = MaxPooling1D(5)(x)
        x = Dropout(0.2)(x)
        x = Conv1D(filters, kernel_size, padding='valid', activation='relu', strides=1)(x)
        x = GlobalMaxPooling1D()(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.2)(x)
        preds = Dense(len(labels_index), activation='softmax')(x)
        model = Model(word_sequence_input, preds)
        model.compile(loss='categorical_crossentropy',
                       optimizer='adam',
                      metrics=['acc'])

    if type == 'cnn_simple_3':
        embedded_input = Dropout(0.2)(word_embedded_sequences)

        y = Conv1D(filters, filter_kernels[0], padding='valid', activation='relu', strides=1)(embedded_input)
        y = MaxPooling1D(5)(y)
        y = Dropout(0.2)(y)


        z = Conv1D(filters, filter_kernels[1], padding='valid', activation='relu', strides=1)(embedded_input)
        z = MaxPooling1D(5)(z)
        z = Dropout(0.2)(z)

        merged = merge([y, z], mode='concat', concat_axis=1)
        x = GlobalMaxPooling1D()(merged)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.2)(x)
        preds = Dense(len(labels_index), activation='softmax')(x)
        model = Model(word_sequence_input, preds)
        model.compile(loss='categorical_crossentropy',
                       optimizer='adam',
                      metrics=['acc'])

    if type == 'char_cnn_2':
        embedded_word = Dropout(0.2)(word_embedded_sequences)
        embedded_char = Dropout(0.2)(TimeDistributed(embedded_char_sequences))


        y = Conv1D(filters, filter_kernels[0], padding='valid', activation='relu', strides=1)(embedded_word)
        y = MaxPooling1D(5)(y)
        y = Dropout(0.2)(y)


        z = Conv1D(filters, filter_kernels[0], padding='valid', activation='relu', strides=1)(embedded_char)
        z = MaxPooling1D(5)(z)
        z = Dropout(0.2)(z)

        merged = merge([z, y], mode='concat', concat_axis=1)
        x = GlobalMaxPooling1D()(merged)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.2)(x)
        preds = Dense(len(labels_index), activation='softmax')(x)

        model = Model([word_sequence_input, char_sequence_input], preds)
        model.compile(loss='categorical_crossentropy',
                        optimizer='adam',
                            metrics=['acc'])




    if type == 'lstm':
        x = Dropout(0.2)(word_embedded_sequences)
        x = Bidirectional(LSTM(64, dropout=0.2, recurrent_dropout=0.2))(x)
        preds = Dense(len(labels_index), activation='softmax')(x)

        model = Model(word_sequence_input, preds)
        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['acc'])

    if type == 'cnn_lstm':
        x = Dropout(0.2)(word_embedded_sequences)
        x = Conv1D(filters, kernel_size, padding='valid', activation='relu', strides=1)(x)
        x = MaxPooling1D(pool_size=pool_size)(x)
        x = Bidirectional(LSTM(128, dropout=0.2, recurrent_dropout=0.2))(x)
        preds = Dense(len(labels_index), activation='softmax')(x)

        model = Model(word_sequence_input, preds)
        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['acc'])

    if type == 'char_cnn':
        x = Dropout(0.2)(word_embedded_sequences)
        x = Conv1D(filters, kernel_size, padding='valid', activation='relu', strides=1)(x)
        x = GlobalMaxPooling1D()(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.2)(x)
        preds = Dense(len(labels_index), activation='softmax')(x)
        model = Model(word_sequence_input, preds)
        model.compile(loss='categorical_crossentropy',
                       optimizer='adam',
                      metrics=['acc'])

    return model