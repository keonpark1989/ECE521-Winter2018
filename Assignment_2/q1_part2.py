import time
import math
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Step 1: read in data from A2 provided files and reshape accordingly
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

# Step 2: define parameters for the model
learning_rate = 0.005   # 0:005 was found to be the optimal learning rate
decay_coef = 0.0
batch_size = 1500        # 500, 1500, 3500
trainData_size = trainData.shape[0]

# n_batches = math.floor(trainData_size/batch_size)
n_batches = math.floor(3500/1500) + 1
# n_epochs = math.floor(20000/n_batches)
n_epochs = math.floor(20000/3)

# Step 3: create placeholders for features and labels
X = tf.placeholder(tf.float64, [None, 784])
Y = tf.placeholder(tf.float64, [None, 1])

# Step 4: create weights and bias
w = tf.Variable(tf.zeros([784, 1], dtype=tf.float64), name = "weights")
b = tf.Variable(tf.zeros([1], dtype=tf.float64), name = "bias")

# Step 5: predict Y from X and w, b
Y_pred = tf.add(tf.matmul(X, w), b)             # note: b is to be broadcasted across the output

# Step 6: define loss function
loss = tf.reduce_sum(tf.square(Y_pred - Y))/(2*tf.cast((tf.shape(X)[0]), dtype=tf.float64))

# Step 7: define training optimizer and MSE + Accuracy Outputs
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
MSE = []
Accuracy = []
Epoch = list(range(1, n_epochs+1))

with tf.Session() as sess:
    # Step 7: initialize the necessary variables, in this case, w and b
    sess.run(tf.global_variables_initializer()) # sess.run(w.initializer) and sess.run(b.initializer)
    # Step 8: train the model
    for i in range(n_epochs):
        for j in range(n_batches):
            if j == n_batches - 1:
                X_batch = np.concatenate((trainData[j*batch_size:trainData_size], trainData[0:((j+1)*batch_size-trainData_size)]), axis=0)
                Y_batch = np.concatenate((trainTarget[j*batch_size:trainData_size], trainTarget[0:((j+1)*batch_size-trainData_size)]), axis=0)
            else:
                X_batch, Y_batch = trainData[j*batch_size:(j+1)*batch_size], trainTarget[j*batch_size:(j+1)*batch_size]
            _, loss_value_batch, Y_pred_batch = sess.run([optimizer, loss, Y_pred], feed_dict = {X: X_batch, Y: Y_batch})
        # Step 9: compute MSE for given epoch pass
        loss_value_set = sess.run(loss, feed_dict = {X: trainData, Y: trainTarget})
        MSE.append(loss_value_set)

print(MSE[-1]) 

# Step 10: plotting the results
plt.plot(Epoch, MSE, '-o', label='Loss value')
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss Value')
plt.legend(ncol=2, loc='upper right')
plt.grid(True)
plt.show()
