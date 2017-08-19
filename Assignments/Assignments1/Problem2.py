# Task 1: Improve the Accuracy of MNIST and Task 2


# Add a hidden layer

import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

import matplotlib.pyplot as plt
import numpy as np
import csv

"""
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

"""


def heart_disease_predict():
    # step 1:read in data and format it
    file = open("./data/heart_disease/heart.csv")
    csvreader = csv.reader(file)
    X = []
    Y = []
    i = 0
    next(csvreader)
    for row in csvreader:
        X.append([])
        Y.append([float(row[9])])
        for j in range(4):
            X[i].append(float(row[j]))

        if row[4] == "Present":
            X[i].append(1.0)
        else:
            X[i].append(0.0)

        for j in range(5, 9):
            X[i].append(float(row[j]))
        i += 1

    train_size = 300
    feature_size = len(X[0])
    batch_size = 10
    hidden_layer = 5

    epoch = 200

    X_train = X[0:train_size]
    Y_train = Y[0:train_size]

    X_test = X[train_size:len(X)]
    Y_test = Y[train_size:len(Y)]

    training_rate = 0.0002
    """
    Build Model
    """
    # step 2: Build placeholder
    X = tf.placeholder(dtype=tf.float32, name="X")
    Y = tf.placeholder(dtype=tf.float32, name="Y")

    # step 3:Build variable
    w1 = tf.Variable(tf.zeros(shape=[feature_size, hidden_layer]), trainable=True, name="weight")
    b1 = tf.Variable(tf.zeros(shape=[1, hidden_layer]), trainable=True, name="bias")

    w2 = tf.Variable(tf.zeros(shape=[hidden_layer, 1]), trainable=True, name="weight")
    b2 = tf.Variable(tf.zeros(shape=[1, 1]), trainable=True, name="bias")

    # step 4:Build predict function
    sig = tf.matmul(X, w1) + b1
    hidden = tf.sigmoid(sig)
    sig2 = tf.matmul(hidden, w2) + b2
    Y_predict=tf.sigmoid(sig2)

    loss = tf.reduce_mean((1-Y)*tf.abs(tf.log(1-Y_predict))+tf.abs(Y*tf.log(Y_predict)))

    # step 5: Build optimizer
    optimizer = tf.train.GradientDescentOptimizer(training_rate).minimize(loss)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        writer = tf.summary.FileWriter("./graphs/Problem3", graph=sess.graph)

        n_batches = train_size / batch_size
        for i in range(epoch):
            total_loss = 0

            for j in range(int(n_batches)):
                start_num = j * batch_size
                _, loss_ = sess.run([optimizer, loss], feed_dict={X: X_train[start_num:start_num + batch_size],
                                                                  Y: Y_train[start_num:start_num + batch_size]})
                total_loss += loss_


            print("Epoch{0}:{1}".format(i, total_loss/n_batches))
            writer.close()

        Y_test_predict = sess.run(Y_predict, feed_dict={X: X_test})
        correct=np.abs(np.around(Y_test_predict)-Y_test)
        accuracy=(len(Y_test)-np.sum(correct))/len(Y_test)
        print(accuracy)



if __name__ == "__main__":
    # ImprovedMNIST()
    heart_disease_predict()
