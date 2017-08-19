#this is the learing notes of TF by szz
#the learning material is Standford Class: Tensorflow for Deep Learning Research

#Author SZZ
#Lesson 1
#Graph and Sessions

import  tensorflow as tf

def example1():
    a=tf.add(3,5)
    sess=tf.Session()
    print(sess.run(a))
    sess.close()

def example2():
    x=2
    y=3
    op1 = tf.add(x,y)
    op2=tf.multiply(x,y)
    op3=tf.pow(op1,op2)

    with tf.Session() as sess:
        op3=sess.run(op3)
        print(op3)

def example3():
    x=2
    y=3
    add_op=tf.add(x,y,name="add")
    mul_op=tf.multiply(x,y,name="mul")
    useless=tf.multiply(x,add_op)
    pow_op=tf.pow(add_op,mul_op)
    with tf.Session() as sess:
        z=sess.run(pow_op) # in this case, the useless node will not be caculated
        print(z)
    with tf.Session() as sess:
        z,not_useless=sess.run([pow_op,useless]) #  in this case, the useless node will be caculated
        print("z=",z,"not_useless=",not_useless)

# graph

def example4():
    g=tf.Graph()
    with g.as_default():
        x=tf.add(3,5)  #  g is setted as default graph, and x is be insert in g
    sess=tf.Session(graph=g)

    print(sess.run(x))
    sess.close()

    # tf.get_default_graph()
    y=tf.add(3,5)
    sess2=tf.Session(graph=tf.get_default_graph())
    print(sess2.run(y))

if __name__=="__main__":
    print("example1:")
    example1()
    print("example2:")
    example2()
    print("example3")
    example3()
    print("example4")
    example4()
