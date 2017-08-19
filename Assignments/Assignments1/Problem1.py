"""
Simple TensorFlow exercises
You should thoroughly test your code
"""

import tensorflow as tf

###############################################################################
# 1a: Create two random 0-d tensors x and y of any distribution.
# Create a TensorFlow object that returns x + y if x > y, and x - y otherwise.
# Hint: look up tf.cond()
# I do the first problem for you
###############################################################################

x = tf.random_uniform([])  # Empty array as shape creates a scalar.
y = tf.random_uniform([])
out = tf.cond(tf.greater(x, y), lambda: tf.add(x, y), lambda: tf.subtract(x, y))

###############################################################################
# 1b: Create two 0-d tensors x and y randomly selected from the range [-1, 1).
# Return x + y if x < y, x - y if x > y, 0 otherwise.
# Hint: Look up tf.case().
###############################################################################

x1 = tf.random_uniform([], -1, 1)
y1 = tf.random_uniform([], -1, 1)
out1 = tf.case({tf.less(x1, y1): lambda: x1 + y1, tf.greater(x1, y1): lambda: x1 - y1},
               default=lambda: tf.constant(0.0), exclusive=True)

###############################################################################
# 1c: Create the tensor x of the value [[0, -2, -1], [0, 1, 2]]
# and y as a tensor of zeros with the same shape as x.
# Return a boolean tensor that yields Trues if x equals y element-wise.
# Hint: Look up tf.equal().
###############################################################################

x2 = tf.constant([[0, -2, -1], [0, 1, 2]])
y2 = tf.zeros_like(x2)
out2 = tf.equal(x2, y2)

###############################################################################
# 1d: Create the tensor x of value
# [29.05088806,  27.61298943,  31.19073486,  29.35532951,
#  30.97266006,  26.67541885,  38.08450317,  20.74983215,
#  34.94445419,  34.45999146,  29.06485367,  36.01657104,
#  27.88236427,  20.56035233,  30.20379066,  29.51215172,
#  33.71149445,  28.59134293,  36.05556488,  28.66994858].
# Get the indices of elements in x whose values are greater than 30.
# Hint: Use tf.where().
# Then extract elements whose values are greater than 30.
# Hint: Use tf.gather().
###############################################################################

x3 = tf.constant([29.05088806, 27.61298943, 31.19073486, 29.35532951,
                  30.97266006, 26.67541885, 38.08450317, 20.74983215,
                  34.94445419, 34.45999146, 29.06485367, 36.01657104,
                  27.88236427, 20.56035233, 30.20379066, 29.51215172,
                  33.71149445, 28.59134293, 36.05556488, 28.66994858])
compare = tf.where(x3 > 30)
elements = tf.gather(x3, compare)

with tf.Session() as sess:
    print(sess.run(compare), sess.run(elements))

###############################################################################
# 1e: Create a diagnoal 2-d tensor of size 6 x 6 with the diagonal values of 1,
# 2, ..., 6
# Hint: Use tf.range() and tf.diag().
###############################################################################

tensor = tf.diag(tf.range(1, 7))
with tf.Session() as sess:
    print(sess.run(tensor))

###############################################################################
# 1f: Create a random 2-d tensor of size 10 x 10 from any distribution.
# Calculate its determinant.
# Hint: Look at tf.matrix_determinant().
###############################################################################

constant = tf.random_normal(shape=[10, 10])
det = tf.matrix_determinant(constant)
with tf.Session() as sess:
    print(sess.run(det))

###############################################################################
# 1g: Create tensor x with value [5, 2, 3, 5, 10, 6, 2, 3, 4, 2, 1, 1, 0, 9].
# Return the unique elements in x
# Hint: use tf.unique(). Keep in mind that tf.unique() returns a tuple.
###############################################################################

constant = tf.constant([5, 2, 3, 5, 10, 6, 2, 3, 4, 2, 1, 1, 0, 9])
uni = tf.unique(constant)
with tf.Session() as sess:
    print(sess.run(uni)[0])

###############################################################################
# 1h: Create two tensors x and y of shape 300 from any normal distribution,
# as long as they are from the same distribution.
# Use tf.cond() to return:
# - The mean squared error of (x - y) if the average of all elements in (x - y)
#   is negative, or
# - The sum of absolute value of all elements in the tensor (x - y) otherwise.
# Hint: see the Huber loss function in the lecture slides 3.
###############################################################################

x_random = tf.random_normal(shape=[300])
y_random = tf.random_normal(shape=[300])
f = tf.cond(tf.less(tf.reduce_mean(x_random - y_random),0), lambda: tf.reduce_mean(tf.square(x_random - y_random)),
            lambda: tf.reduce_sum(tf.abs(x_random - y_random)))
with tf.Session() as sess:
    print(sess.run(f))
