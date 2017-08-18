# Author:SZZ
# Lesson 3

# Linear Regression and MNIST project

import tensorflow as tf
import xlrd
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials import mnist as mn


def Lineary_Regression():
    # step 1 :read in data from xls file
    DATA_FILE = './data/fire_theft.xls'

    book = xlrd.open_workbook(DATA_FILE, encoding_override='utf-8')
    sheet = book.sheet_by_index(0)
    data = np.asarray([sheet.row_values(i) for i in range(1, sheet.nrows)])
    n_sample = sheet.nrows - 1

    # step 2: Build placeholder for X,Y
    X = tf.placeholder(dtype=tf.float32, name="X")
    Y = tf.placeholder(dtype=tf.float32, name="Y")

    # step 3: Build training variable
    weight = tf.Variable(0.0, trainable=True, name="weight")
    bias = tf.Variable(0.0, trainable=True, name="bias")

    # step 4: Build prediction function
    Y_predict = X * weight + bias  # this is  an add op

    # step 5: Build loss function
    loss = tf.square(Y - Y_predict, name="loss_function")

    # step 6: Build optimizer
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss)

    # Run Session
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        writer = tf.summary.FileWriter('./graphs/linear_reg', sess.graph)

        for i in range(50):
            total_loss = 0

            for x, y in data:
                _, l = sess.run([optimizer, loss], feed_dict={X: x, Y: y})
                total_loss += l
            print("Epoch {0}:{1}".format(i, total_loss / n_sample))
        writer.close()

        weight, bias = sess.run([weight, bias])

    # plot results

    X, Y = data.T[0], data.T[1]
    print(weight, bias)
    print(X * weight + bias)
    plt.plot(X, X * weight + bias, 'r', label='Predict data')
    plt.plot(X, Y, 'bo', label='Real data')
    plt.legend()
    plt.show()


def logistic_reg():
    # constant
    batch_size = 128
    image_size = 784
    learning_rate = 0.01
    epoch = 30

    # read in data
    data = mn.input_data.read_data_sets('./data/mnist', one_hot=True)

    # define placeholder
    X = tf.placeholder(dtype=tf.float32, shape=[batch_size, image_size], name="X")
    Y = tf.placeholder(dtype=tf.float32, shape=[batch_size, 10], name="Y")

    # define training variable
    w = tf.Variable(tf.random_normal(shape=[image_size, 10], stddev=0.01,dtype=tf.float32), trainable=True)
    b = tf.Variable(tf.zeros(shape=[1, 10]), trainable=True)

    # define model
    logits = tf.matmul(X, w) + b

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

                #very important, don not use the same name. such as  _,
                # _,loss = sess.run([optimizer, loss], feed_dict={X: x_batches, Y: y_batches})
                # this is two loss, will run feed in wrong data
                _, loss_batch = sess.run([optimizer, loss], feed_dict={X: x_batches, Y: y_batches})
                total_loss += loss_batch
            print("Epoch{0}:{1}".format(i, total_loss/batch_size))

        writer.close()

        print("Optimize finished")

        # Caculate Accuracy

        Y_predict=tf.nn.softmax(logits=logits)
        correct=tf.equal(tf.arg_max(Y_predict,1),tf.arg_max(Y,1))
        accuracy=tf.reduce_sum(tf.cast(correct,tf.float32))


        total_correct=0
        for _ in range(n_batches):
            x_batches, y_batches = data.train.next_batch(batch_size)
            accuracy_num = sess.run(accuracy, feed_dict={X: x_batches, Y: y_batches})
            total_correct += accuracy_num

        print("Accuracy:{0}".format(total_correct/data.train.num_examples ))

if __name__ == "__main__":
    # Lineary_Regression()
    logistic_reg()
