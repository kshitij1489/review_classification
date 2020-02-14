import os
import tensorflow as tf
from tensorflow.python.keras.optimizers import Adam

import numpy as np

device = '/cpu:0'

class AttentionLSTM(tf.keras.Model):
    def __init__(self, hidden_size, num_classes):
        super(AttentionLSTM, self).__init__()

        H_LSTM, sequence_length, num_classes = 64, 100, 8
        
        initializer = tf.initializers.VarianceScaling(scale=2.0)
        
        self.embedding = tf.keras.layers.Embedding(input_dim=50000, output_dim=50, input_length=sequence_length)
        
        self.bi_lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(H_LSTM, return_sequences=True), merge_mode ='sum')
        
        self.W = tf.Variable(create_matrix_with_kaiming_normal((H_LSTM, 1)))
        
        self.dropout = tf.keras.layers.Dropout(0.5)
        
        self.fc_out = tf.keras.layers.Dense(num_classes, activation='softmax',
                                            kernel_initializer=initializer)
    
    def call(self, input_tensor, training=False):

        x = self.embedding(input_tensor) # N, 100, 50
        
        H = self.bi_lstm(x) # N, 100, 64
        
        M = tf.nn.tanh(H) # N, 100, 64
        M = tf.reshape(M, [-1, 64])
        
        WM = tf.matmul(M, self.W)
        WM = tf.reshape(WM, [-1, 100]) # N, 100

        alpha = tf.nn.softmax(WM)
        #alpha = tf.reshape(alpha, [-1, 100, 1]) # N, 100, 1
        r = tf.matmul(tf.transpose(H, [0, 2, 1]), tf.reshape(alpha, [-1, 100, 1])) # N, 64, 1
        #r = tf.matmul(H, alpha)
        r = tf.squeeze(r) # N, 64
        
        H_star = tf.nn.tanh(r)
        
        H_star = self.dropout(H_star, training=training)
        
        out = self.fc_out(H_star) # N, 8
        
        return out

def train_part(model, train_dset, val_dset, num_epochs=1, is_training=False, print_every = 100):

    with tf.device(device):
        
        train_loss = tf.keras.metrics.Mean(name='train_loss')
        train_accuracy = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')
    
        val_loss = tf.keras.metrics.Mean(name='val_loss')
        val_accuracy = tf.keras.metrics.CategoricalAccuracy(name='val_accuracy')
        
        loss_fn = tf.keras.losses.CategoricalCrossentropy()
        optimizer = tf.keras.optimizers.Adam(1e-3)
        
        t = 0
        for epoch in range(num_epochs):
            
            # Reset the metrics - https://www.tensorflow.org/alpha/guide/migration_guide#new-style_metrics
            train_loss.reset_states()
            train_accuracy.reset_states()
            
            for x_np, y_np in train_dset:
                # Use the model function to build the forward pass.
                loss_value, scores, grads = grad(model, loss_fn, x_np, y_np)
                optimizer.apply_gradients(zip(grads, model.trainable_variables))

                # Update the metrics
                train_loss.update_state(loss_value)
                train_accuracy.update_state(y_np, scores)

                if t % print_every == 0:
                    val_loss.reset_states()
                    val_accuracy.reset_states()
                    for test_x, test_y in val_dset:
                        # During validation at end of epoch, training set to False
                        t_loss, prediction = loss(model, loss_fn, test_x, test_y, training=False)

                        val_loss.update_state(t_loss)
                        val_accuracy.update_state(test_y, prediction)

                    template = 'Iteration {}, Epoch {}, Loss: {}, Accuracy: {}, Val Loss: {}, Val Accuracy: {}'
                    print (template.format(t, epoch+1,train_loss.result(), train_accuracy.result()*100,
                                           val_loss.result(),val_accuracy.result()*100))
                t += 1


def grad(model, loss_fn, inputs, targets):
    with tf.GradientTape() as tape:
        loss_value, score = loss(model, loss_fn, inputs, targets, training=True)
    return loss_value, score, tape.gradient(loss_value, model.trainable_variables)

def loss(model, loss_fn, x, y, training):
    y_ = model(x, training=training)
    return loss_fn(y_true=y, y_pred=y_), y_

def validation_test(model, val_dataset):
    test_accuracy = tf.keras.metrics.CategoricalAccuracy()
    loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    for (x, y) in val_dataset:
        # training=False is needed only if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        prediction = model(x, training=False)
        loss = loss_object(y, prediction)
        test_accuracy(prediction, y)

    print("Test set accuracy: {:.3%}".format(test_accuracy.result()))
    
def evaluate(model, test_dataset):
    test_accuracy = tf.keras.metrics.CategoricalAccuracy()
    for (x, y) in test_dataset:
        # training=False is needed only if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        logits = model(x, training=False)
        prediction = tf.argmax(logits, axis=1, output_type=tf.int32)
        print(y[0], logits[0])
        test_accuracy(y, prediction)

    print("Test set accuracy: {:.3%}".format(test_accuracy.result()))    
                    
                    
def create_matrix_with_kaiming_normal(shape):
    if len(shape) == 2:
        fan_in, fan_out = shape[0], shape[1]
    elif len(shape) == 4:
        fan_in, fan_out = np.prod(shape[:3]), shape[3]
    return tf.keras.backend.random_normal(shape) * np.sqrt(2.0 / fan_in)


class Dataset(object):
    def __init__(self, X, y, batch_size, shuffle=False):
        """
        Construct a Dataset object to iterate over data X and labels y
        
        Inputs:
        - X: Numpy array of data, of any shape
        - y: Numpy array of labels, of any shape but with y.shape[0] == X.shape[0]
        - batch_size: Integer giving number of elements per minibatch
        - shuffle: (optional) Boolean, whether to shuffle the data on each epoch
        """
        assert X.shape[0] == y.shape[0], 'Got different numbers of data and labels'
        self.X, self.y = X, y
        self.batch_size, self.shuffle = batch_size, shuffle

    def __iter__(self):
        N, B = self.X.shape[0], self.batch_size
        idxs = np.arange(N)
        if self.shuffle:
            np.random.shuffle(idxs)
        return iter((self.X[i:i+B], self.y[i:i+B]) for i in range(0, N, B))