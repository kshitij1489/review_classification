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
    
    def __init__(self, tokenizer, label_index, max_word_count = 50000, max_sequence_len = 100, word_embedding_dim = 50):
        self.max_word_count = max_word_count
        self.max_sequence_len = max_sequence_len
        self.embedding_dim = word_embedding_dim
        self.tokenizer = tokenizer
        self.label_index = label_index
        self.model_lstm = None
        self.model_cnn = None
        self.model_mlp = None

    def train_LSTM(self, X_train, y_train, epochs = 5, batch_size = 64, learning_rate = 0.001, reg = 0.01):
        """
        Trains LSTM
        - X_train: Input sequence
        - y_train: Target sequence
        - epochs
        - batch_size
        - learning_rate = Adam optimizer's learning rate
        - reg: Regularization
        Returns :
        - history: Scalar loss
        """

        flatten_y = [category for sublist in y_train for category in sublist]
        class_weights = class_weight.compute_class_weight('balanced', np.unique(flatten_y), flatten_y)
        optim = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        
        model = models.Sequential()
        model.add(Embedding(self.max_word_count, self.embedding_dim))
        model.add(SpatialDropout1D(0.2))
        model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(8, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer=optim, metrics=['categorical_accuracy'])

        history = model.fit(X_train, y_train, class_weight = class_weight, epochs = epochs,
                    batch_size=batch_size, validation_split=0.1,
                    callbacks=[EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)])
        
        self.model_lstm = model
        return history

    def train_LSTM_1(self, X_train, y_train, epochs = 5, batch_size = 64, learning_rate = 0.001, reg = 0.01):
        """
        Trains LSTM
        - X_train: Input sequence
        - y_train: Target sequence
        - epochs
        - batch_size
        - learning_rate = Adam optimizer's learning rate
        - reg: Regularization
        Returns :
        - history: Scalar loss
        """
        flatten_y = [category for sublist in y_train for category in sublist]
        class_weights = class_weight.compute_class_weight('balanced', np.unique(flatten_y), flatten_y)
        optim = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        
        model = models.Sequential()
        model.add(Embedding(self.max_word_count, 64))
        model.add(Bidirectional(LSTM(64, return_sequences=True)))
        model.add(Bidirectional(LSTM(32)))
        model.add(Dense(64, kernel_regularizer=regularizers.l2(reg), activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(8, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer=optim, metrics=['categorical_accuracy'])
        
        history = model.fit(X_train, y_train, class_weight = class_weight, epochs = epochs,
            batch_size=batch_size, validation_split=0.1,
            callbacks=[EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)])

        self.model_lstm = model
        return history
    
    def train_CNN(self, X_train, y_train, epochs = 5, batch_size = 64, learning_rate = 0.001, regularization = 0.01):
        """
        Trains CNN
        - X_train: Input sequence
        - y_train: Target sequence
        - epochs
        - batch_size
        - learning_rate = Adam optimizer's learning rate
        - reg: Regularization
        Returns :
        - history: Scalar loss
        """
        flatten_y = [category for sublist in y_train for category in sublist]
        class_weights = class_weight.compute_class_weight('balanced', np.unique(flatten_y), flatten_y)
        optim = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        model = models.Sequential()
        model.add(Embedding(self.max_word_count, self.embedding_dim))
        model.add(Dropout(rate=0.1))
        model.add(Conv1D(300, 3, padding='valid', activation='relu', strides=1))
        model.add(GlobalMaxPool1D())
        model.add(Dense(8, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer=optim, metrics=['categorical_accuracy'])
        history = model.fit(X_train, y_train, class_weight = class_weight, epochs = epochs,
                            batch_size=batch_size, validation_split=0.1,
                            callbacks=[EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)])

        self.model_cnn = model
        return history
    
    def train_MLP(self, X_train, y_train, epochs = 5, batch_size = 64, learning_rate = 0.001, reg = 0.01):
        """
        Trains LSTM
        - X_train: Input sequence
        - y_train: Target sequence
        - epochs
        - batch_size
        - learning_rate = Adam optimizer's learning rate
        - reg: Regularization
        Returns :
        - history: Scalar loss
        """        
        flatten_y = [category for sublist in y_train for category in sublist]
        class_weights = class_weight.compute_class_weight('balanced', np.unique(flatten_y), flatten_y)
        optim = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        model = models.Sequential()
        model.add(Dense(512, activation='relu', kernel_regularizer=regularizers.l2(reg), input_shape=(self.max_sequence_len,)))
        model.add(Dropout(0.2))
        for _ in range(3):
            model.add(Dense(256, activation='relu'))
            model.add(Dropout(0.2))
        model.add(Dense(8, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer=optim, metrics=['categorical_accuracy'])
        history = model.fit(X_train, y_train, class_weight = class_weight, epochs = epochs,
                            batch_size=batch_size, validation_split=0.1,
                            callbacks=[EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)])
        self.model_mlp = model
        return history    
    
    def predict(self, review, model_name):
        """
        This function takes in a list of sentences as input, performs binary classification for each 
        of the categories, and selects a sentence for prediction if it's confidence is > 0.5 for a
        particular category.
        - review: list of sentences to be classified
        - model_name: Model type
        Returns :
        - prediction_list: A list of important sentences in the form of (confidence, category, sentence) 
        """
        seq = self.tokenizer.texts_to_sequences(review)
        padded = pad_sequences(seq, maxlen = self.max_sequence_len)
        
        if model_name == 'LSTM':
            pred = self.model_lstm.predict(padded)
        elif model_name == 'CNN':
            pred = self.model_cnn.predict(padded)
        elif model_name == 'MLP':
            pred = self.model_mlp.predict(padded)
        else:
            raise ValueError('Model not defined')

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
    
    def evaluate(self, X_test, y_test, model_name):
        if model_name == 'LSTM':
            test_loss, test_acc = self.model_lstm.evaluate(X_test,y_test)
        elif model_name == 'CNN':
            test_loss, test_acc = self.model_cnn.evaluate(X_test,y_test)
        elif model_name == 'MLP':
            test_loss, test_acc = self.model_mlp.evaluate(X_test,y_test)
        else:
            raise ValueError('Model not defined')

        return test_loss, test_acc
    
    def train(self, X_train, y_train, model_name, epochs = 5, batch_size = 64,
                   learning_rate = 0.001, regularization = 0.01):
        if model_name == 'LSTM':
            history = self.train_LSTM(X_train, y_train, epochs, batch_size, learning_rate, regularization)
        elif model_name == 'CNN':
            history = self.train_CNN(X_train, y_train, epochs, batch_size, learning_rate, regularization)
        elif model_name == 'MLP':
            history = self.train_MLP(X_train, y_train, epochs, batch_size, learning_rate, regularization)
        else:
            raise ValueError('Model not defined')
        
        return history
        

def tokenize_data(X, y, max_word_count=50000, max_sequence_len=100):

    tokenizer = Tokenizer(num_words=max_word_count, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
    tokenizer.fit_on_texts(X)
    word_index = tokenizer.word_index

    X_tokenized = tokenizer.texts_to_sequences(X)
    X_tokenized = pad_sequences(X_tokenized, maxlen = max_sequence_len)

    mlb = MultiLabelBinarizer()
    y_tokenized = mlb.fit_transform(y)

    label_index = mlb.classes_

    print('Shape of data tensor:', X.shape)
    print('Found %s unique tokens.' % len(word_index))

    return X_tokenized, y_tokenized, tokenizer, label_index

def split_to_sentences(paragraph):
    return tokenize.sent_tokenize(paragraph)

def clean_data(text):
    text = text.lower().strip()
    text = text.strip(".")
    # Remove double space
    # Trim input?
    return text

def extract_text_blocks(review):
    """
    This functions splits the string into sentence, then does basic text cleansing
    and flattens the list of sentences.
    - review: Review in the form of a string
    Returns :
    - flat_list: A flattened list of sentences 
    """
    sentences = split_to_sentences(review)
    text_block = [sentence.split(",") for sentence in sentences]
    flat_list = [clean_data(block) for sublist in text_block for block in sublist]
    return flat_list