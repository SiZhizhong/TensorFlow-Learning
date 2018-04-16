import xlrd
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as  np

DATA_FILE = "./data/fire_theft.xls"
# data preparation
book = xlrd.open_workbook(DATA_FILE, encoding_override='utf-8')
sheet = book.sheet_by_index(0)
data = np.asarray([sheet.row_values(i) for i in range(1, sheet.nrows)])
n_sample = sheet.nrows - 1

# graph
x = tf.placeholder(tf.float32, shape=[None, 1], name="x")
y = tf.placeholder(tf.float32, shape=[None, 1], name='y')

W = tf.get_variable("W", shape=[1, 1], dtype=tf.float32, initializer=tf.random_normal_initializer(0, 0.1))
b = tf.get_variable('b', shape=[1, 1], dtype=tf.float32, initializer=tf.zeros_initializer)

Y_predict = W * x + b

loss = tf.reduce_mean(tf.square(Y_predict - y))

optimizer = tf.train.GradientDescentOptimizer(0.001).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    writer = tf.summary.FileWriter('./graphs', sess.graph)
    for epoch in range(100):

        _,l=sess.run([optimizer,loss],feed_dict={x:data[0:,0].reshape(-1,1),y:data[0:,1].reshape(-1,1)})
        print("Epoch:",epoch," loss:", l)
    W_1,b_1=sess.run([W,b])


print(W_1,b_1)
fig=plt.figure()
axs=fig.add_subplot(111)
axs.scatter(data[0:,0],data[0:,1])
plt.plot(data[0:,0],np.squeeze(data[0:,0]*W_1+b_1),"r-",lw=5)
plt.show()
