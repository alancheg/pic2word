# CNN_test.py

# from tensorflow.python.ops import rnn
from tensorflow.contrib.rnn.python.ops import core_rnn_cell_impl, rnn, rnn_cell
import tensorflow as tf
import numpy as np
# from tensorflow.python.ops.rnn_cell import
# from tensorflow.python.ops.rnn_cell import MultiRNNCell
# import gensim.models.word2vec


class RNN:

    def __init__(self):
        # 定义在类中方便了之后的调用
        self.session = tf.Session()
        self.input_layer = None
        self.label_layer = None
        self.weights = None
        self.biases = None
        self.lstm_cell = None
        self.prediction = None
        self.loss = None
        self.trainer = None

    def _del_(self):
        self.session.close()

    def train(self, train_x, train_y, learning_rate=0.05, limit=1000, batch_n=1, seq_len=3, input_n=2, hidden_n=5, output_n=2):
        # RNN 训练
        # 输入层
        self.input_layer = [tf.placeholder("float", [seq_len, input_n]) for i in range(batch_n)]

        # 标签
        self.label_layer = tf.placeholder("float", [seq_len, output_n])

        self.weights = tf.Variable(tf.random_normal([hidden_n, output_n]))
        self.biases = tf.Variable(tf.random_normal([output_n]))

        # self.lstm_cell = rnn_cell.BasicLSTMCell(hidden_n, forget_bias=1.0)
        # 设置 神经元的个数 和 forget_bias 参数
        self.lstm_cell = core_rnn_cell_impl.BasicLSTMCell(hidden_n, forget_bias=1.0)

        # outputs, states = rnn.rnn(self.lstm_cell, self.input_layer, dtype=tf.float32)
        outputs, states = tf.contrib.rnn.static_rnn(self.lstm_cell, self.input_layer, dtype=tf.float32)

        self.prediction = tf.matmul(outputs[-1], self.weights) + self.biases

        # 定义 损失函数
        # self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.prediction, self.label_layer))
        #todo:need fix
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.predict(test_x=train_x), logits=self.label_layer))

        # 使用 Adam 优化器进行训练
        self.trainer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.loss)

        initer = tf.initialize_all_variables()

        train_x = train_x.reshape((batch_n, seq_len, input_n))
        train_y = train_y.reshape((seq_len, output_n))

        # run graph
        self.session.run(initer)
        for i in range(limit):
            self.session.run(self.trainer, feed_dict={self.input_layer[0]: train_x[0], self.label_layer: train_y})

    def predict(self, test_x):
        return self.session.run(self.prediction, feed_dict={self.input_layer[0]: test_x})

    def test(self):
        # 测试函数
        train_x = np.array([[1, 2, 3, 4, 5, 6]])
        train_y = np.array([[1, 2, 3, 4, 5, 6]])

        self.train(train_x, train_y, batch_n=1, seq_len=3, input_n=2)
        test_x = np.array([[1, 1], [2, 2], [3, 3]])
        # test_x = train_x.reshape((1, 3, 2))[0]
        print(self.predict(test_x))


if __name__ == "__main__":
    nn = RNN()
    nn.test()