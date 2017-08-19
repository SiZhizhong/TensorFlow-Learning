# Task 1: Improve the Accuracy of MNIST and Task 2


# Add a hidden layer

import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

import matplotlib.pyplot as plt
import numpy as np


# Add A hidden layer
def ImprovedMNIST():
    # constant
    batch_size = 128
    image_size = 784
    learning_rate = 0.01
    epoch = 30
    hidden_size = 100

    # read in data
    # Task 1
    data = input_data.read_data_sets('./data/MNIST', one_hot=True)
    # Task 2
    # data = input_data.read_data_sets('../notMNIST-to-MNIST', one_hot=True)

    # define placeholder
    X = tf.placeholder(dtype=tf.float32, shape=[batch_size, image_size], name="X")
    Y = tf.placeholder(dtype=tf.float32, shape=[batch_size, 10], name="Y")

    # define training variable
    w1 = tf.Variable(tf.random_normal(shape=[image_size, hidden_size], stddev=0.01, dtype=tf.float32), trainable=True)
    b1 = tf.Variable(tf.zeros(shape=[1, hidden_size]), trainable=True)

    w2 = tf.Variable(tf.random_normal(shape=[hidden_size, 10], stddev=0.01, dtype=tf.float32), trainable=True)
    b2 = tf.Variable(tf.zeros(shape=[1, 10]), trainable=True)

    # define model

    hidden_layer = tf.sigmoid(tf.matmul(X, w1) + b1)

    logits = tf.matmul(hidden_layer, w2) + b2

    # define loss function
    entropy = tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=logits)
    loss = tf.reduce_mean(entropy)

    # define optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

    # run session
    with tf.Session() as sess:
        # initialize all variables
        sess.run(tf.global_variables_initializer())
        # graph
        writer = tf.summary.FileWriter("./graphs/Note3", graph=sess.graph)

        n_batches = int(data.train.num_examples / batch_size)
        # strat training
        for i in range(epoch):
            total_loss = 0
            for _ in range(n_batches):
                x_batches, y_batches = data.train.next_batch(batch_size)

                # very important, don not use the same name. such as  _,
                # _,loss = sess.run([optimizer, loss], feed_dict={X: x_batches, Y: y_batches})
                # this is two loss, will run feed in wrong data
                _, loss_batch = sess.run([optimizer, loss], feed_dict={X: x_batches, Y: y_batches})
                total_loss += loss_batch
            print("Epoch{0}:{1}".format(i, total_loss / batch_size))

        writer.close()

        print("Optimize finished")

        # Caculate Accuracy

        Y_predict = tf.nn.softmax(logits=logits)
        correct = tf.equal(tf.arg_max(Y_predict, 1), tf.arg_max(Y, 1))
        accuracy = tf.reduce_sum(tf.cast(correct, tf.float32))

        total_correct = 0
        for _ in range(n_batches):
            x_batches, y_batches = data.train.next_batch(batch_size)
            accuracy_num = sess.run(accuracy, feed_dict={X: x_batches, Y: y_batches})
            total_correct += accuracy_num

        print("Accuracy:{0}".format(total_correct / data.train.num_examples))

import  xlrd
def heart_disease_predict():
    book=xlrd.open_workbook()


if __name__ == "__main__":
    #ImprovedMNIST()
    heart_disease_predict()
