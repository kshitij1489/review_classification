import os
import tensorflow as tf
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
import helper

import numpy as np

device = '/cpu:0'

class AttentionLSTM(tf.keras.Model):
    def __init__(self, sequence_length, embedding_dim):
        super(AttentionLSTM, self).__init__()

        self.H_LSTM, self.sequence_length, num_classes = 64, sequence_length, 8
        
        initializer = tf.initializers.VarianceScaling(scale=2.0)
        
        self.embedding = tf.keras.layers.Embedding(input_dim=50000, output_dim=embedding_dim, input_length=self.sequence_length)
        
        self.bi_lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(self.H_LSTM, return_sequences=True), merge_mode ='sum')
        
        self.W = tf.Variable(create_matrix_with_kaiming_normal((self.H_LSTM, 1)))
        
        self.dropout = tf.keras.layers.Dropout(0.5)
        
        self.fc_out = tf.keras.layers.Dense(num_classes, activation='sigmoid',
                                            kernel_initializer=initializer)
    
    def call(self, input_tensor, training=False):

        x = self.embedding(input_tensor) # N, 100, 50
        
        H = self.bi_lstm(x) # N, 100, 64
        
        M = tf.nn.tanh(H) # N, 100, 64
        M = tf.reshape(M, [-1, self.H_LSTM])
        
        WM = tf.matmul(M, self.W)
        WM = tf.reshape(WM, [-1, self.sequence_length]) # N, 100

        alpha = tf.nn.softmax(WM)

        r = tf.matmul(tf.transpose(H, [0, 2, 1]), tf.reshape(alpha, [-1, self.sequence_length, 1])) # N, 64, 1

        r = tf.squeeze(r) # N, 64
        
        H_star = tf.nn.tanh(r)
        
        H_star = self.dropout(H_star, training=training)
        
        out = self.fc_out(H_star) # N, 8
        
        return out

def train_part(model, train_dset, val_dset, num_epochs=1, is_training=False):
    
    history = {'loss' : [], 'val_loss' : [], 'accuracy' : [], 'val_accuracy' : []}
    
    with tf.device(device):
        
        train_loss = tf.keras.metrics.Mean(name='train_loss')
        train_accuracy = tf.keras.metrics.BinaryAccuracy(name='train_accuracy')
    
        val_loss = tf.keras.metrics.Mean(name='val_loss')
        val_accuracy = tf.keras.metrics.BinaryAccuracy(name='val_accuracy')
        
        loss_fn = tf.keras.losses.BinaryCrossentropy()
        optimizer = tf.keras.optimizers.Adam(1e-3)
        
        for epoch in range(num_epochs):
            
            # Reset the metrics - https://www.tensorflow.org/alpha/guide/migration_guide#new-style_metrics
            train_loss.reset_states()
            train_accuracy.reset_states()
            val_loss.reset_states()
            val_accuracy.reset_states()
            
            # Train for one epoch
            for x_np, y_np in train_dset:
                # Use the model function to build the forward pass.
                loss_value, scores, grads = grad(model, loss_fn, x_np, y_np)
                optimizer.apply_gradients(zip(grads, model.trainable_variables))
                
                # Update the metrics
                train_loss.update_state(loss_value)
                train_accuracy.update_state(y_np, scores)
                
            # Evaluate validation set after one epoch
            for test_x, test_y in val_dset:
                # During validation at end of epoch, training set to False
                t_loss, prediction = loss(model, loss_fn, test_x, test_y, training=False)

                val_loss.update_state(t_loss)
                val_accuracy.update_state(test_y, prediction)

            history['loss'].append(train_loss.result().numpy())
            history['val_loss'].append(val_loss.result().numpy())
            history['accuracy'].append(train_accuracy.result().numpy())
            history['val_accuracy'].append(val_accuracy.result().numpy())
                
            template = 'Epoch {}, Loss: {}, Accuracy: {}, Val Loss: {}, Val Accuracy: {}'
            print (template.format(epoch+1,train_loss.result(), train_accuracy.result()*100,
                                   val_loss.result(), val_accuracy.result()*100))
            
    return history


def grad(model, loss_fn, inputs, targets):
    with tf.GradientTape() as tape:
        loss_value, score = loss(model, loss_fn, inputs, targets, training=True)
    return loss_value, score, tape.gradient(loss_value, model.trainable_variables)

def loss(model, loss_fn, x, y, training):
    y_ = model(x, training=training)
    return loss_fn(y_true=y, y_pred=y_), y_
    
def evaluate(model, test_dataset):
    
    test_accuracy = tf.keras.metrics.BinaryAccuracy()
    test_loss = tf.keras.metrics.Mean()
    loss_fn = tf.keras.losses.BinaryCrossentropy()
    
    for (x, y) in test_dataset:
        t_loss, prediction = loss(model, loss_fn, x, y, training=False)
        test_accuracy.update_state(y, prediction)
        test_loss.update_state(t_loss)

    return test_loss.result().numpy() , test_accuracy.result().numpy()
                    
def create_matrix_with_kaiming_normal(shape):
    if len(shape) == 2:
        fan_in, fan_out = shape[0], shape[1]
    elif len(shape) == 4:
        fan_in, fan_out = np.prod(shape[:3]), shape[3]
    return tf.keras.backend.random_normal(shape) * np.sqrt(2.0 / fan_in)

def predict_labels(review, model, tokenizer, label_index):
    seq = tokenizer.texts_to_sequences(review)
    padded = pad_sequences(seq, maxlen = 100)
    pred = helper.predict(model, padded)
    
    filtered_indices = np.where(np.array(pred) > 0.5)
    i_review, j_label = filtered_indices[0], filtered_indices[1]
    
    prediction_list = helper.return_labels(i_review, j_label, pred, review, label_index)
    return prediction_list
