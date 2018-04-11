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
validData = validData.reshape([validData.shape[0], 28*28])
testData = testData.reshape([testData.shape[0], 28*28])

# set parameters for the model
learning_rate = 0.005
batch_size = 500 
decay_coef = 0.001
trainData_size = trainData.shape[0]
n_batches = math.floor(trainData_size/batch_size)
n_epochs = math.floor(20000/n_batches)

# create placeholders for features and labels 
X = tf.placeholder(tf.float64, [None, 784])
Y = tf.placeholder(tf.float64, [None, 1])

# create weights and bias variables for training
w = tf.Variable(tf.zeros([784, 1], dtype=tf.float64), name = "weights")
b = tf.Variable(tf.zeros([1], dtype=tf.float64), name = "bias")

# predicting Y with linear regression
Y_pred = tf.add(tf.matmul(X, w), b)           # note: b is to be broadcasted across the output

# formulate loss (MSE) expression
loss = tf.reduce_sum(tf.square(Y_pred - Y))/(2*tf.cast((tf.shape(X)[0]), dtype=tf.float64)) + 0.5*decay_coef*tf.reduce_sum(tf.square(w))

# define training optimizer and MSE + Accuracy Outputs
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
MSE = []
accuracy_validset = 0
Epoch = list(range(1, n_epochs+1))

with tf.Session() as sess:
    # initialize the necessary variables: w (weights) and b (bias)
    sess.run(tf.global_variables_initializer())
    
    # apply SGD optimizer over the minibatches for each epoch pass
    for i in range(n_epochs):
        for j in range(n_batches):
            X_batch, Y_batch = trainData[j*batch_size:(j+1)*batch_size], trainTarget[j*batch_size:(j+1)*batch_size]
            sess.run(optimizer, feed_dict = {X: X_batch, Y: Y_batch})            
        # compute MSE for given epoch pass
        loss_value_set = sess.run(loss, feed_dict = {X: trainData, Y: trainTarget})
        MSE.append(loss_value_set)
    
    # compute validation accuracy after training complete
    Y_pred_lr = sess.run(Y_pred, feed_dict = {X: validData, Y: validTarget})
    Y_pred_set = tf.sign(tf.sign(Y_pred_lr-0.5)+1)
    Y_pred_bool = tf.equal(Y_pred_set, validTarget)
    accuracy_validset = sess.run(tf.reduce_sum(tf.cast(Y_pred_bool, tf.float64))/(validData.shape[0]))
    
print(MSE[-1])
print(accuracy_validset)

# Step 10: plotting the results
plt.plot(Epoch, MSE, label='Loss value')
plt.title('Training Loss: Lambda = 0.001')
plt.xlabel('Epoch')
plt.ylabel('Loss Value')
plt.legend(ncol=2, loc='upper right')
plt.grid(True)
plt.show()
