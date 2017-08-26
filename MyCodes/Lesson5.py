# Rewrite the MNIST model by object orient

from TFModel import TFModel
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import numpy


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
            feed_dict = {self.X: input_batch}
        else:
            feed_dict = {self.X: input_batch, self.Y: label_batch}
        return feed_dict

    def add_training_op(self, loss_op):
        self.global_step = tf.Variable(0, trainable=False, dtype=tf.int32, name="global_step")
        self.optimizer = tf.train.GradientDescentOptimizer(self.config.training_rate).minimize(loss_op,
                                                                                               global_step=self.global_step)

    def run_epoch(self, sess, input_data=None, input_labels=None):
        n_batch = int(self.all_data.train.num_examples / self.config.batch_size)
        total_loss = 0
        for _ in range(n_batch):
            x_batch, y_batch = self.all_data.train.next_batch(self.config.batch_size)
            feed_dict = self.create_feed_dict(x_batch, y_batch)
            _, loss = sess.run([self.optimizer, self.loss_op], feed_dict=feed_dict)
            total_loss += loss
        average_loss = total_loss / n_batch
        return average_loss

    def fit(self, sess, input_data=None, input_labels=None):

        saver = tf.train.Saver(max_to_keep=5)

        #sess.run(tf.global_variables_initializer())
        self.writer = tf.summary.FileWriter('./graphs/mnst', graph=sess.graph)
        loss_array = []
        for epoch in range(self.config.epochs):
            epoch_loss = self.run_epoch(sess, input_data, input_labels)
            loss_array.append([epoch, epoch_loss])
            saver.save(sess, './checkpoint/MNIST', global_step=self.global_step)

        return loss_array

    def predict(self, sess, input_data, input_labels=None):
        # read in trained model variables,return the number
        logits=sess.run(self.out_layer_logitst,feed_dict={self.X:input_data})
        num=sess.run(tf.nn.softmax(logits))
        predict_num=numpy.argmax(num)
        return predict_num
        # if not trained ,return None

    def restore(self, sess):
        ckpt = tf.train.get_checkpoint_state('./checkpoint/')
        saver = tf.train.Saver()
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)

    def caculate_accuracy(self,sess):
        Y_predict = tf.nn.softmax(logits=self.out_layer_logitst)
        correct = tf.equal(tf.arg_max(Y_predict, 1), tf.arg_max(self.Y, 1))
        accuracy = tf.reduce_sum(tf.cast(correct, tf.float32))
        n_batches = int(self.all_data.train.num_examples / self.config.batch_size)
        total_correct = 0
        for _ in range(n_batches):
            x_batches, y_batches = self.all_data.train.next_batch(self.config.batch_size)
            accuracy_num = sess.run(accuracy, feed_dict={self.X: x_batches,self. Y: y_batches})
            total_correct += accuracy_num

        print("Accuracy:{0}".format(total_correct /self.all_data.train.num_examples ))

    def __init__(self, Config,sess):
        # constant settings
        self.config = Config
        # read in data
        # self.load_data()
        # add placeholder
        self.add_placeholders()
        # create model
        self.add_model()
        # add loss op
        self.add_loss_op(self.Y)
        self.add_training_op(self.loss_op)
        sess.run(tf.global_variables_initializer())



class Config(object):
    training_rate = 0.1
    epochs = 30
    batch_size = 128


if __name__ == "__main__":
    sess = tf.Session()
    my_mnist_model = MNISTModel(Config(),sess)
    my_mnist_model.load_data()


    my_mnist_model.restore(sess)
    #loss_array=my_mnist_model.fit(sess)
    #print(loss_array)
    #my_mnist_model.caculate_accuracy(sess)




    x_batch,y_batch=my_mnist_model.all_data.train.next_batch(1000)
    for i in range(5):
        number=numpy.random.randint(0,1000-1)

        data=numpy.reshape(x_batch[number],[1,784])
        predict_num=my_mnist_model.predict(sess,data)
        print("predict_num {0},actual_num {1}".format(predict_num,numpy.argmax(y_batch[number])))

        plt.imshow(numpy.reshape(data,[28,28]))
        plt.show()




