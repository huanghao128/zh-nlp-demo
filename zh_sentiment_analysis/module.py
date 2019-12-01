import tensorflow as tf
from numpy.random import seed
seed(1)


class RNNSentimentModel():

    def __init__(self, max_seq_length, vocab_size, embedding_size, hidden_size, layer_size, n_class, learning_rate):
        self.max_seq_length = max_seq_length
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.layer_size = layer_size
        self.n_class = n_class
        self.learning_rate = learning_rate

        self.create_graph()

    def unit_lstm(self):
        # 定义一层 LSTM_Cell
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(num_units=self.hidden_size, forget_bias=1.0, state_is_tuple=True)
        return lstm_cell

    def create_graph(self):
        with tf.name_scope('Input_Layer'):
            self.input_x = tf.placeholder(tf.int32, [None, self.max_seq_length], name='input_x')
            self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
            self.label_y = tf.placeholder(tf.float32, [None, self.n_class], name='label_y')

        # Embedding layer
        with tf.name_scope('Embedding_layer'):
            embeddings_var = tf.Variable(tf.random_normal([self.vocab_size, self.embedding_size]), dtype=tf.float32)
            # embeddings_var = tf.Variable(self.word_embedding, trainable=True, dtype=tf.float32)
            batch_embedded = tf.nn.embedding_lookup(embeddings_var, self.input_x)

        with tf.name_scope('RNN_layer'):
            mlstm_cell = tf.contrib.rnn.MultiRNNCell([self.unit_lstm() for i in range(self.layer_size)], state_is_tuple=True)
            outputs, state = tf.nn.dynamic_rnn(mlstm_cell, inputs=batch_embedded, time_major=False, dtype=tf.float32)
            outputs = tf.transpose(outputs, [1, 0, 2])

        with tf.name_scope('Dropout_layer'):
            drop_outputs = tf.nn.dropout(outputs[-1], keep_prob=self.keep_prob)

        with tf.name_scope('Output_layer'):
            W = tf.Variable(tf.random_normal([self.hidden_size, self.n_class]), dtype=tf.float32)
            b = tf.Variable(tf.random_normal([self.n_class]), dtype=tf.float32)
            y_outputs = tf.matmul(drop_outputs, W) + b

        with tf.name_scope('Loss'):
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=y_outputs, labels=self.label_y))
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

        with tf.name_scope('Accuracy'):
            self.score = tf.nn.softmax(y_outputs, name="score")
            self.predictions = tf.argmax(y_outputs, 1, name="predictions")
            self.correct_pred = tf.equal(tf.argmax(y_outputs, 1), tf.argmax(self.label_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))