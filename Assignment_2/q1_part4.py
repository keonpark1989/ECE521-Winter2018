import time
import math
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# read in data from A2 provided files and reshape accordingly
def loadData():
    with np.load("notMNIST.npz") as data :
        Data, Target = data ["images"], data["labels"]
        posClass = 2
        negClass = 9
        dataIndx = (Target==posClass) + (Target==negClass)
        Data = Data[dataIndx]/255.
        Target = Target[dataIndx].reshape(-1, 1)
        Target[Target==posClass] = 1
        Target[Target==negClass] = 0
        np.random.seed(521)
        randIndx = np.arange(len(Data))
        np.random.shuffle(randIndx)
        Data, Target = Data[randIndx], Target[randIndx]
        trainData, trainTarget = Data[:3500], Target[:3500]
        validData, validTarget = Data[3500:3600], Target[3500:3600]
        testData, testTarget = Data[3600:], Target[3600:]
        
    return trainData, trainTarget, validData, validTarget, testData, testTarget


(trainData, trainTarget, validData, validTarget, testData, testTarget) = loadData()

# resize the input image format
trainData = trainData.reshape([trainData.shape[0], 28*28])

# set parameters and adjust for bias term
x_bias = tf.ones([trainData.shape[0],1])
trainDataMatrix = tf.concat([x_bias, trainData], 1)

t1 = time.time()
# implement normal equations
w = tf.matmul(tf.matmul(tf.matrix_inverse(tf.matmul(tf.transpose(trainDataMatrix), trainDataMatrix)), tf.transpose(trainDataMatrix)), tf.cast(trainTarget, tf.float32))

# compute MSE for training dataset
Y_pred = tf.matmul(trainDataMatrix, w)                #do not have to broadcast bias as it is included in both tensors
MSE = tf.reduce_sum(tf.square(Y_pred - trainTarget))/(2*tf.cast((tf.shape(trainDataMatrix)[0]), dtype=tf.float32))

# compute accuracy for training dataset
Y_pred_set = tf.sign(tf.sign(Y_pred-0.5)+1)
Y_pred_bool = tf.equal(Y_pred_set, trainTarget)
accuracy = tf.reduce_sum(tf.cast(Y_pred_bool, tf.float32))/(trainTarget.shape[0])

# compute time difference
t2 = time.time()
diff_time = t2 - t1

# compute the least squares solution
with tf.Session() as sess:
    #print(sess.run(trainDataMatrix))
    #print(w) #display weight parameters (bias, w1, w2, ..., wd)
    
    print(sess.run(MSE))
    print(sess.run(accuracy))
    print(diff_time)