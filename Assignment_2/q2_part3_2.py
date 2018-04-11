import time
import math
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

N = [i / 1000.0 for i in range(1000)]
Y_pred = tf.constant(N)

Y = tf.zeros([1000], dtype=tf.float32)


# Logistic Regression Loss Function
#cross_entropy_error = tf.nn.sigmoid_cross_entropy_with_logits(labels=Y, logits=Y_pred)
cross_entropy_error = -tf.multiply(Y, tf.log(Y_pred)) - tf.multiply((tf.ones([1000]) - Y), tf.log(tf.ones([1000]) - Y_pred))
# Linear Regression Loss Function
mean_square_error = tf.square(Y_pred - Y)

with tf.Session() as sess:
    cee = sess.run(cross_entropy_error)
    mse = sess.run(mean_square_error)

plt.plot(N, cee, '-', label='Cross Entropy Loss')
plt.plot(N, mse, '-', label='Mean Squared Loss')

plt.title('Cross Entropy Loss vs Mean Squared Loss')
plt.xlabel('Y - Y_pred')
plt.ylabel('Loss Value')
plt.legend(ncol=1, loc='upper left')
plt.grid(True)
plt.show()
