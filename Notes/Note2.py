# Author SZZ
# Lesson 2
# TensorBoard, Constant, Variable and Placeholder


import tensorflow as tf


# TensorBoard: visualize the model
def example1():
    a = tf.constant(2, name="a")
    b = tf.constant(3, name="b")
    node = tf.add(a, b, name="add")

    # with tf.Session(graph=tf.get_default_graph()) as sess:
    with tf.Session() as sess:  # it will use the default graph as the graph parameter
        writer = tf.summary.FileWriter(".\graphs", sess.graph)
        print(sess.run(node))

    writer.close()


# Constants
def example2():
    a = tf.constant([2, 2])
    b = tf.constant([[0, 1], [2, 3]])
    x = tf.add(a, b)  # add a to each row of b
    y = tf.multiply(a, b)  # nultiply a with each row of b
    with tf.Session() as sess:
        x, y = sess.run([x, y])
        print(x, '\n', y)
        print(sess.graph.as_graph_def())

    temp1 = tf.zeros([2, 3], tf.int32)  # return a tensor with size of (2,3) and each element is 0
    temp2 = tf.zeros_like(temp1)  # return a tensor with size is same as temp1

    temp3 = tf.ones([3, 5], tf.float32)
    temp4 = tf.ones_like(temp3)

    temp5 = tf.linspace(0.0, 10.0, 11)  # start and end must be float
    temp6 = tf.range(0, 10, 2)

    # random

    temp7 = tf.random_normal([3, 3], 0, 1)
    temp8 = tf.random_uniform([3, 3], 0, 10)

    with tf.Session() as sess2:
        a, b, c, d, e, f, g, h = sess2.run([temp1, temp2, temp3, temp4, temp5, temp6, temp7, temp8])
        print(a, b, c, d, e, f, g, h)


# variables
def example3():
    va1 = tf.Variable(2, name="scalar")
    va2 = tf.Variable([2, 3], name="vector")
    va3 = tf.Variable(tf.zeros([784, 10]), name="matrix")

    # initialize is imperative
    # global initialize
    init = tf.global_variables_initializer()
    # initialize a subset of variable
    # init=tf.variables_initializer([va1,va2],name="initilize_va1_and_va2")
    with tf.Session() as sess:
        sess.run(init)
        # sess.run(va3.initilizer)
        print(va1.eval(), va2.eval(), va3.eval())

    va4 = tf.Variable(10)
    va4.assign(100)
    va5 = tf.Variable(2)
    with tf.Session() as sess2:
        sess2.run(va4.initializer)
        print(va4.eval())  # will return 10, because va4.assign(100) is operation , but it doesn't run
        sess2.run(va5.assign(100))  # assign will call initializer automatically
        print(va5.eval())
        sess2.run(va5.assign_add(100))
        print(va5.eval())
        sess2.run(va5.assign_sub(10))
        print(va5.eval())

    # each session maintain its own copy of variable

    va6 = tf.Variable(10)
    sess3 = tf.Session()
    sess4 = tf.Session()
    sess3.run(va6.initializer)
    sess4.run(va6.initializer)

    print(sess3.run(va6.assign_add(10)))
    print(sess4.run(va6.assign_sub(10)))

    # use variable to initilize another variable
    w1 = tf.Variable(tf.random_normal([10, 10], 0, 1))
    w2 = tf.Variable(w1.initialized_value())  # should use initialized value

    tf.InteractiveSession().run(w2.initializer)
    print(w2.eval())


# Placeholder
def example4():
    a = tf.placeholder(tf.float32, shape=[3])
    b = tf.constant([1, 2, 3], dtype=tf.float32)
    c = a + b
    with tf.Session() as sess:
        print(sess.run(c, {a: [1, 2, 3]}))

    # Not only placeholder is feedable
    a = tf.add(2, 5)
    b = tf.multiply(a, 3)
    with tf.Session() as sess2:
        print(sess2.run(b, feed_dict={a: 3}))


if __name__ == "__main__":
    print("example1")
    example1()
    print('example2')
    example2()
    print("example3")
    example3()
    print("example4")
    example4()
