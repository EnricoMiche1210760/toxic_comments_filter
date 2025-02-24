import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv1D, Bidirectional, MaxPooling1D, Flatten, Embedding,\
      LSTM, Dense, Dropout, GlobalMaxPool1D, GRU, SimpleRNN, Dropout, BatchNormalization
from tensorflow.keras import Sequential

class EarlyStopping(tf.keras.callbacks.Callback):
    '''
    Stop training when a monitored quantity has stopped improving.
    '''
    def on_epoch_end(self, epoch, logs):
        '''
        Called at the end of an epoch. 
        Args:
            epoch: Integer, index of epoch.
            logs: Dict, metric results for this training epoch, and for the validation epoch if validation is performed.
        '''
        if logs['val_accuracy'] > 0.97 and epoch > 3:
            self.model.stop_training = True
            print(f'\nStop training at epoch: {epoch+1}\n')

class BatchLogger(tf.keras.callbacks.Callback):
    '''
    Callback that logs metrics at the end of each batch.
    '''
    def on_batch_end(self, batch_number, logs):
        '''
        Called at the end of a batch.
        Args:
            batch_number: Integer, index of batch within the current epoch.
            logs: Dict, metric results for this batch.
        '''
        if batch_number % 150 == 0:
            self.batch_loss.append(logs['loss'])
            self.batch_accuracy.append(logs['accuracy'])
    def on_train_begin(self, *args, **kwargs):
        '''
        Called at the beginning of training.
        Defines the attributes to store the loss and accuracy of each batch.
        '''
        self.batch_loss = []
        self.batch_accuracy = []


class F1Score(tf.keras.metrics.Metric):
    '''
    Computes the F1 Score.
    '''
    def __init__(self, name='f1_score', **kwargs):
        '''
        Initializes the instance attributes.
        Args:
            name: String, name of the metric instance.
            **kwargs: Arbitrary keyword arguments.
        '''
        super(F1Score, self).__init__(name=name, **kwargs)
        self.precision = tf.keras.metrics.Precision()
        self.recall = tf.keras.metrics.Recall()

    def update_state(self, y_true, y_pred, sample_weight=None):
        '''
        Accumulates the confusion matrix.
        Args:
            y_true: Ground truth values. shape = [batch_size, d0, .. dN].
            y_pred: The predicted values. shape = [batch_size, d0, .. dN].
            sample_weight: Optional sample_weight acts as a coefficient for the loss. shape = [batch_size].
        '''
        self.precision.update_state(y_true, y_pred, sample_weight)
        self.recall.update_state(y_true, y_pred, sample_weight)

    def result(self):
        '''
        Computes and returns the F1 Score.
        Returns:
            F1 Score: Float.
        '''
        precision = self.precision.result()
        recall = self.recall.result()
        return 2 * ((precision * recall) / (precision + recall + tf.keras.backend.epsilon()))

    def reset_states(self):
        '''
        Resets all of the metric state variables.
        '''
        self.precision.reset_states()
        self.recall.reset_states()



@keras.saving.register_keras_serializable()#senza specificare un paramentro, sovvrascrive loss
def weighted_binary_crossentropy(w0, w1):
    '''
    Computes the weighted binary crossentropy loss.
    Args:
        w0: Float, weight for the positive class.
        w1: Float, weight for the negative class.
    Returns:
        loss: Function, weighted binary crossentropy loss function.
    '''
    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1-1e-7)
        _w0 = tf.constant(w0, dtype=tf.float32)
        _w1 = tf.constant(w1, dtype=tf.float32)
        loss = -_w1 * y_true * tf.math.log(y_pred) - _w0 * (1 - y_true) * tf.math.log(1 - y_pred)
        return tf.reduce_mean(loss)
    return loss


def LSTM_model(vocab_size, maxlen, dense_units=1, activation='softmax'):
    '''
    LSTM model
    Args:
        vocab_size: int, vocabulary size
        maxlen: int, maximum length of the sequences
        dense_units: int, number of units in the dense layer
        activation: str, activation function for the output layer
    Returns:
        model: keras.Sequential, LSTM model
    '''
    model = Sequential()
    model.add(Embedding(input_dim = vocab_size, output_dim=150, input_length=maxlen))
    model.add(Dropout(0.65))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(GlobalMaxPool1D())
    model.add(Dropout(0.35))
    model.add(Dense(dense_units, activation=activation))
    return model

def GRU_RNN_model(vocab_size, maxlen, dense_units=1, activation='softmax'):
    '''
    GRU RNN model
    Args:
        vocab_size: int, vocabulary size
        maxlen: int, maximum length of the sequences
        dense_units: int, number of units in the dense layer
        activation: str, activation function for the output layer
    Returns:
        model: keras.Sequential, GRU RNN model
    '''
    model = Sequential()
    model.add(Embedding(input_dim = vocab_size, output_dim=100, input_length=maxlen))
    model.add(Dropout(0.85))
    model.add(GRU(units=100, return_sequences=True))
    model.add(Dropout(0.5))
    model.add(SimpleRNN(units=50))
    model.add(Dropout(0.35))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.15))
    model.add(Dense(dense_units, activation=activation))
    return model

def CNN_model(vocab_size, maxlen, dense_units=1, activation='softmax'):
    '''
    CNN model
    Args:
        vocab_size: int, vocabulary size
        maxlen: int, maximum length of the sequences
        dense_units: int, number of units in the dense layer
        activation: str, activation function for the output layer
    Returns:
        model: keras.Sequential, CNN model
    '''
    model = Sequential()
    model.add(Embedding(input_dim = vocab_size, output_dim=150, input_length=maxlen))
    model.add(Dropout(0.75))
    model.add(Bidirectional(LSTM(units=100, return_sequences=True)))
    model.add(Dropout(0.5))
    model.add(Conv1D(64, 5, activation='relu'))
    model.add(GlobalMaxPool1D())
    model.add(Dropout(0.35))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(dense_units, activation=activation))
    return model

def CNN_model_2(vocab_size, maxlen, dense_units=1, activation='softmax'):
    '''
    CNN model
    Args:
        vocab_size: int, vocabulary size
        maxlen: int, maximum length of the sequences
        dense_units: int, number of units in the dense layer
        activation: str, activation function for the output layer
    Returns:
        model: keras.Sequential, CNN model
    '''
    model = Sequential()
    model.add(Embedding(input_dim = vocab_size, output_dim=256, input_length=maxlen))
    model.add(Dropout(0.8))
    model.add(Conv1D(filters=64, kernel_size=3, padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))
    model.add(LSTM(units=64))
    model.add(Dropout(0.6))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(dense_units, activation=activation))
    return model