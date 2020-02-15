import tensorflow as tf
import numpy as np
import pandas as pd
import pickle
from nltk import tokenize
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.utils import class_weight

from tensorflow.python.keras import models, initializers, regularizers
from tensorflow.python.keras.metrics import BinaryAccuracy, Accuracy
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.layers import Dense, Conv1D, Dropout, Embedding, LSTM, SpatialDropout1D, GlobalMaxPool1D, Bidirectional

from tensorflow.python.keras.callbacks import EarlyStopping
from helper import load_glove_embedding, return_labels, Dataset
from lstm_attention_model import AttentionLSTM, train_part, evaluate, predict_labels

_embedding_index_glove = None

class TextClassifier:
    
    def __init__(self, tokenizer, label_index, max_word_count = 50000, max_sequence_len = 100,
                 word_embedding_dim = 50, verbose=1):
        self.max_word_count = max_word_count
        self.max_sequence_len = max_sequence_len
        self.embedding_dim = word_embedding_dim
        self.tokenizer = tokenizer
        self.label_index = label_index
        self.history = None
        self.model = None
        self.embedding_matrix_glove = None
        self.verbose = verbose

    def _train_LSTM(self, X_train, y_train, epochs = 5, batch_size = 64, learning_rate = 0.001, reg = 0.01):
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
        model.add(Embedding(input_dim=self.max_word_count, output_dim=self.embedding_dim))
        model.add(SpatialDropout1D(0.2))
        model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(8, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer=optim, metrics=[BinaryAccuracy()])

        history = model.fit(X_train, y_train, class_weight = class_weight, epochs = epochs,
                    batch_size=batch_size, validation_split=0.25, verbose = self.verbose,
                    callbacks=[EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)])
        
        self.model = model
        self.history = history.history

    def _train_LSTM_1(self, X_train, y_train, epochs = 5, batch_size = 64, learning_rate = 0.001, reg = 0.01):
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
        model.compile(loss='binary_crossentropy', optimizer=optim, metrics=[BinaryAccuracy()])
        
        history = model.fit(X_train, y_train, class_weight = class_weight, epochs = epochs,
            batch_size=batch_size, validation_split=0.25, verbose = self.verbose,
            callbacks=[EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)])

        self.model = model
        self.history = history.history
    
    def _train_CNN(self, X_train, y_train, epochs = 5, batch_size = 64, learning_rate = 0.001, regularization = 0.01):
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
        model.add(Embedding(input_dim=self.max_word_count, output_dim=self.embedding_dim)) 
        model.add(Conv1D(filters=300, kernel_size=3, padding='valid', activation='relu', strides=1)) 
        model.add(GlobalMaxPool1D())
        model.add(Dense(8, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer=optim, metrics=[BinaryAccuracy()])
        history = model.fit(X_train, y_train, class_weight = class_weight, epochs = epochs,
                            batch_size=batch_size, validation_split=0.25, verbose = self.verbose,
                            callbacks=[EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)])

        self.model = model
        self.history = history.history
    
    def _train_CNN_Glove(self, X_train, y_train, epochs = 5, batch_size = 64, learning_rate = 0.001, regularization = 0.01):
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
        
        embedding_matrix = self.create_embedding_matrix()

        model = models.Sequential()
        model.add(Embedding(input_dim=self.max_word_count, output_dim=100,
                            embeddings_initializer = initializers.Constant(embedding_matrix),
                            input_length=self.max_sequence_len, trainable=False))
        model.add(Conv1D(filters=300, kernel_size=3, padding='valid', activation='relu', strides=1))
        model.add(GlobalMaxPool1D())
        model.add(Dense(8, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer=optim, metrics=[BinaryAccuracy()])
        history = model.fit(X_train, y_train, class_weight = class_weight, epochs = epochs,
                            batch_size=batch_size, validation_split=0.25, verbose = self.verbose,
                            callbacks=[EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)])

        self.model = model
        self.history = history.history
    
    def _train_CNN_Glove_1(self, X_train, y_train, epochs = 5, batch_size = 64, learning_rate = 0.001, regularization = 0.01):
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
        
        embedding_matrix = self.create_embedding_matrix()

        model = models.Sequential()
        model.add(Embedding(self.max_word_count, 100, embeddings_initializer = initializers.Constant(embedding_matrix),
                            input_length=self.max_sequence_len, trainable=False))
        model.add(Conv1D(filters=128, kernel_size=5, padding='valid', activation='relu', strides=1))
        model.add(Conv1D(filters=256, kernel_size=3, padding='valid', activation='relu', strides=1))
        model.add(GlobalMaxPool1D())
        model.add(Dense(8, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer=optim, metrics=[BinaryAccuracy()])
        history = model.fit(X_train, y_train, class_weight = class_weight, epochs = epochs,
                            batch_size=batch_size, validation_split=0.25,
                            callbacks=[EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)])

        self.model = model
        self.history = history.history

    
    def _train_MLP(self, X_train, y_train, epochs = 5, batch_size = 64, learning_rate = 0.001, reg = 0.01):
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
        model.compile(loss='binary_crossentropy', optimizer=optim, metrics=[BinaryAccuracy()])
        history = model.fit(X_train, y_train, class_weight = class_weight, epochs = epochs,
                            batch_size=batch_size, validation_split=0.25, verbose = self.verbose,
                            callbacks=[EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)])

        
        self.model = model
        self.history = history.history
        
    def _train_Attention(self, X_train, y_train, epochs = 5, batch_size = 64, learning_rate = 0.001, reg = 0.01):
        """
        Trains BI-LSTM with Attention. Reference: https://www.aclweb.org/anthology/P16-2034/
        Note: this is model has a different api and was included here to be compatible with the cross-validation
        code in the notebook.
        - X_train: Input sequence
        - y_train: Target sequence
        - epochs
        - batch_size
        - learning_rate = Adam optimizer's learning rate
        - reg: Regularization
        Returns :
        - history: Scalar loss
        """
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1)
        # Split data into batches of 64
        train_dset = Dataset(X_train, y_train, batch_size=batch_size, shuffle=True)
        val_dset = Dataset(X_val, y_val, batch_size=batch_size, shuffle=True)
        
        model = AttentionLSTM(self.max_sequence_len, self.embedding_dim)
        
        history = train_part(model, train_dset, val_dset,num_epochs=10,
                             is_training=True, learning_rate=learning_rate, verbose=self.verbose)
        
        # Update the key name for consistency.
        history['binary_accuracy'] = history.pop('accuracy')
        history['val_binary_accuracy'] = history.pop('val_accuracy')
        
        self.model = model
        self.history = history
        

    def train(self, X_train, y_train, model_name, epochs = 5, batch_size = 64,
                   learning_rate = 0.001, regularization = 0.01):
        """
        An api for training the model
        - X_train: Input sequence
        - y_train: Target sequence
        - epochs
        - batch_size
        - learning_rate = Adam optimizer's learning rate
        - reg: Regularization
        Returns :
        - history: Scalar loss
        """        
        if model_name == 'LSTM':
            self._train_LSTM(X_train, y_train, epochs, batch_size, learning_rate, regularization)
        elif model_name == 'CNN':
            self._train_CNN(X_train, y_train, epochs, batch_size, learning_rate, regularization)
        elif model_name == 'MLP':
            self._train_MLP(X_train, y_train, epochs, batch_size, learning_rate, regularization)
        elif model_name == 'CNN_Glove':
            self._train_CNN_Glove(X_train, y_train, epochs, batch_size, learning_rate, regularization)
        elif model_name == 'Attention':
            self._train_Attention(X_train, y_train, epochs, batch_size, learning_rate, regularization)  
        else:
            raise ValueError('Model not defined')

        return self.history
    
    def predict(self, review):
        """
        This function takes in a list of sentences as input, performs binary classification for each 
        of the categories, and selects a sentence for prediction if it's confidence is > 0.5 for a
        particular category.
        
        Note: The Attention model has a different api and won't be supported here.
        
        - review: list of sentences to be classified
        - model_name: Model type
        Returns :
        - prediction_list: A list of important sentences in the form of (confidence, category, sentence) 
        """
        seq = self.tokenizer.texts_to_sequences(review)
        padded = pad_sequences(seq, maxlen = self.max_sequence_len)
        
        pred = self.model.predict(padded)
        
        filtered_indices = np.where(np.array(pred) > 0.5)
        i_review, j_label = filtered_indices[0], filtered_indices[1]

        prediction_list = return_labels(i_review, j_label, pred, review, self.label_index)

        return prediction_list
    
    def evaluate(self, X_test, X_test):
        """
        Model evaluation on the test set. It uses the binary accuracy metric to compute the score
        
        Note: The Attention model has a different api and won't be supported here.
        
        - X_test: 
        - X_test: 
        Returns :
        - test_loss: Binary cross entropy loss 
        - test_acc: binary accuracy score
        """ 
        test_loss, test_acc = self.model.evaluate(X_test,y_test)

        return test_loss, test_acc
    
    def create_embedding_matrix(self):
        """
        Model evaluation on the test set. It uses the binary accuracy metric to compute the score
        
        Note: The Attention model has a different api and won't be supported here.
        
        - X_test: 
        - X_test: 
        Returns :
        - test_loss: Binary cross entropy loss 
        - test_acc: binary accuracy score
        """         
        self.load_glove()
        
        embedding_matrix = np.zeros((50000, 100))
        for word, index in self.tokenizer.word_index.items():
            if index > 50000 - 1:
                break
            else:
                if word in _embedding_index_glove:
                    embedding_matrix[index] = _embedding_index_glove[word]
                    
        return embedding_matrix
    
    def load_glove(self):
        """
        This function loads the glove embedding values in the global variable.
        It avoids reloading the same instance over and over.
        """         
        global _embedding_index_glove
        
        if _embedding_index_glove is None:
            _embedding_index_glove =  load_glove_embedding()

def tokenize_data(X, y, max_word_count=50000, max_sequence_len=100):
    """
    This function tokenizes the text blocks, pads the tokenized text blocks to 
    have uniform length.
    - X: list of strings
    - y: list of strings
    - max_word_count: 
    - max_sequence_len:
    Returns :
    - X_tokenized: Tokenized X
    - y_tokenized: Tokenized y
    - tokenizer: tokenizer object, which contains all the mapping of words and indices
    - label_index: index to word mapping, a dict
    """
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
    """
    This function uses the nltk library to split a document into sentences
    - paragraph: String
    Returns :
    - tokenized: list of sentences
    """
    return tokenize.sent_tokenize(paragraph)

def clean_data(text):
    """
    This function cleans up the string by removing trailing spaces and punctutations.
    - text: String
    Returns :
    - text: 
    """
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
    #text_block = [sentence.split(",") for sentence in sentences]
    #flat_list = [clean_data(block) for sublist in text_block for block in sublist]
    #return flat_list
    return sentences
    