# Rewrite the MNIST model by object orient

from TFModel import TFModel
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


class MNISTModel(TFModel):
    def load_data(self):
        self.all_data = input_data.read_data_sets('./data/mnist', one_hot=True)

    def add_placeholders(self):
        # for input data
        self.X = tf.placeholder(dtype=tf.float32, shape=[None, 784], name="X")
        # for input label
        self.Y = tf.placeholder(dtype=tf.float32, shape=[None, 10], name="Y")

    def add_model(self, input_data=None):
        self.hidden_layer_size = 100
        self.weight1 = tf.Variable(tf.random_normal(shape=[784, self.hidden_layer_size]), trainable=True,
                                   name="weight1")
        self.bias1 = tf.Variable(tf.random_normal(shape=[1, self.hidden_layer_size]), trainable=True, name="bias1")

        self.weight2 = tf.Variable(tf.random_normal(shape=[self.hidden_layer_size, 10]), trainable=True, name="weight2")
        self.bias2 = tf.Variable(tf.random_normal(shape=[1, 10]), trainable=True, name="bias2")

        self.hidden_layer_logits = tf.matmul(self.X, self.weight1) + self.bias1
        self.hidden_layer_output = tf.sigmoid(self.hidden_layer_logits)

        self.out_layer_logitst = tf.matmul(self.hidden_layer_output, self.weight2) + self.bias2
        return self.out_layer_logitst

    def add_loss_op(self, pred):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=pred, logits=self.out_layer_logitst)
        self.loss_op = tf.reduce_mean(cross_entropy)
        return self.loss_op

    def create_feed_dict(self, input_batch, label_batch=None):
        if label_batch == None:
            feed_dict = {self.X: input_data}
        else:
            feed_dict = {self.X: input_data, self.Y: label_batch}
        return feed_dict

    def add_training_op(self, loss_op):
        self.optimizer = tf.train.GradientDescentOptimizer(self.config.training_rate).minimize(loss_op)

    def run_epoch(self, sess, input_data, input_labels):
        n_batch = int(self.all_data.train.num_examples / self.config.batch_size)
        total_loss = 0
        for _ in range(n_batch):
            x_batch, y_batch = self.all_data.train.next_batch(self.config.batch_size)
            feed_dict = self.create_feed_dict(x_batch, y_batch)

            _, loss = sess.run([self.optimizer, self.loss_op], feed_dict=feed_dict)
            total_loss += loss
        average_loss = total_loss / n_batch
        return average_loss


def fit(self, sess, input_data, input_labels):
    sess.run(tf.global_variables_initializer())
    self.writer = tf.summary.FileWriter('./graphs/mnst', graph=sess.graph)
    loss_array = []
    for epoch in range(self.config.epochs):
        epoch_loss = self.run_epoch(input_data, input_labels)
        loss_array.append([epoch, epoch_loss])
    return loss_array


def __init__(self, Config):
    self.config = Config


class Config(object):
    training_rate = 0.1
    epochs = 30
    batch_size = 128
