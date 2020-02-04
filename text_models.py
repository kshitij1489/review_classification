import tensorflow as tf
import numpy as np
import pandas as pd
import pickle
from nltk import tokenize
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.utils import class_weight

from tensorflow.python.keras import models, initializers, regularizers
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.layers import Dense, Conv1D, Dropout, Embedding, LSTM, SpatialDropout1D, GlobalMaxPool1D, Bidirectional

from tensorflow.python.keras.callbacks import EarlyStopping

class TextClassifier:
    
    def __init__(self, max_word_count = 50000, max_sequence_len = 100, word_embedding_dim = 50):
        self.max_word_count = max_word_count
        self.max_sequence_len = max_sequence_len
        self.embedding_dim = word_embedding_dim
        self.tokenizer = None
        self.label_index = None
        self.model_lstm = None
        self.model_cnn = None
        self.model_mlp = None

    def train_LSTM_1(self, X_train, y_train, epochs = 5, batch_size = 64):

        model = models.Sequential()
        model.add(Embedding(self.max_word_count, self.embedding_dim))
        model.add(SpatialDropout1D(rate=0.2))
        model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(8, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])

        history = model.fit(X_train, y_train, epochs = epochs,
                    batch_size=batch_size, validation_split=0.1,
                    callbacks=[EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)])
        
        self.model_lstm = model
        return history

    def train_LSTM(self, X_train, y_train, epochs = 5, batch_size = 64):

        model = models.Sequential()
        model.add(Embedding(self.max_word_count, 64))
        model.add(Bidirectional(LSTM(64, return_sequences=True)))
        model.add(Bidirectional(LSTM(32)))
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(8, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])
        
        history = model.fit(X_train, y_train, epochs = epochs,
            batch_size=batch_size, validation_split=0.1,
            callbacks=[EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)])

        self.model_lstm = model
        return history
    
    def train_CNN(self, X_train, y_train, epochs = 5, batch_size = 64):
        
        flatten_y = [category for sublist in y_train for category in sublist]
        class_weights = class_weight.compute_class_weight('balanced', np.unique(flatten_y), flatten_y)

        model = models.Sequential()
        model.add(Embedding(self.max_word_count, self.embedding_dim))
        model.add(Dropout(rate=0.1))
        model.add(Conv1D(300, 3, padding='valid', activation='relu', strides=1))
        model.add(GlobalMaxPool1D())
        model.add(Dense(8, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])
        history = model.fit(X_train, y_train, class_weight = class_weight, epochs = epochs,
                            batch_size=batch_size, validation_split=0.1,
                            callbacks=[EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)])

        self.model_cnn = model
        return history
    
    def train_MLP(self, X_train, y_train, epochs = 5, batch_size = 64):
        
        flatten_y = [category for sublist in y_train for category in sublist]
        class_weights = class_weight.compute_class_weight('balanced', np.unique(flatten_y), flatten_y)

        model = models.Sequential()
        model.add(Dense(512, activation='relu', input_shape=(self.max_sequence_len,)))
        model.add(Dropout(0.2))
        for _ in range(3):
            model.add(Dense(256, activation='relu'))
            model.add(Dropout(0.2))
        model.add(Dense(8, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])
        history = model.fit(X_train, y_train, class_weight = class_weight, epochs = epochs,
                            batch_size=batch_size, validation_split=0.1,
                            callbacks=[EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)])
        self.model_mlp = model
        return history    
    
    def tokenize_data(self, X, y):
        
        self.tokenizer = Tokenizer(num_words=self.max_word_count, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
        self.tokenizer.fit_on_texts(X)
        word_index = self.tokenizer.word_index
        
        X_tokenized = self.tokenizer.texts_to_sequences(X)
        X_tokenized = pad_sequences(X_tokenized, maxlen = self.max_sequence_len)

        mlb = MultiLabelBinarizer()
        y_tokenized = mlb.fit_transform(y)
        
        self.label_index = mlb.classes_
        
        print('Shape of data tensor:', X.shape)
        print('Found %s unique tokens.' % len(word_index))
        
        return X_tokenized, y_tokenized
    
    def predict(self, review, model_name):

        seq = self.tokenizer.texts_to_sequences(review)
        padded = pad_sequences(seq, maxlen = self.max_sequence_len)
        if model_name == 'LSTM':
            pred = self.model_lstm.predict(padded)
        elif model_name == 'CNN':
            pred = self.model_cnn.predict(padded)

        pred_id = np.argmax(pred, axis=1)
        conf = pred[np.arange(len(review)), pred_id]
        conf[conf < 0.5] = 0

        prediction_list = self.return_labels(conf, pred_id, review)

        return prediction_list

    def return_labels(self, conf, pred_id, review):
        prediction_list = []
        for c,i, block in zip(conf, pred_id, review):
            if c != 0:
                prediction_list.append((c, self.label_index[i], block))

        return prediction_list
    

def split_to_sentences(paragraph):
    return tokenize.sent_tokenize(paragraph)

def clean_data(text):
    text = text.lower().strip()
    text = text.strip(".")
    # Remove double space
    # Trim input?
    return text

def extract_text_blocks(review):
    sentences = split_to_sentences(review)
    text_block = [sentence.split(",") for sentence in sentences]
    flat_list = [clean_data(block) for sublist in text_block for block in sublist]
    return flat_list