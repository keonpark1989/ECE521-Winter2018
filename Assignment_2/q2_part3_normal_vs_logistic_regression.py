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

#NORMAL EQUATIONS FOR LEAST SQUARE SOLUTION
# set parameters and adjust for bias term
x_bias_train = tf.ones([trainData.shape[0],1])
x_bias_valid = tf.ones([validData.shape[0],1])
x_bias_test = tf.ones([testData.shape[0],1])
trainDataMatrix = tf.concat([x_bias_train, trainData], 1)
validDataMatrix = tf.concat([x_bias_valid, validData], 1)
testDataMatrix = tf.concat([x_bias_test, testData], 1)

# implement normal equations
w_ls = tf.matmul(tf.matmul(tf.matrix_inverse(tf.matmul(tf.transpose(trainDataMatrix), trainDataMatrix)), tf.transpose(trainDataMatrix)), tf.cast(trainTarget, tf.float32))

# compute MSE for training dataset
Y_pred_train = tf.matmul(trainDataMatrix, w_ls)
Y_pred_valid = tf.matmul(validDataMatrix, w_ls)
Y_pred_test = tf.matmul(testDataMatrix, w_ls)

# compute accuracy for training dataset
Y_pred_train_set = tf.sign(tf.sign(Y_pred_train-0.5)+1)
Y_pred_valid_set = tf.sign(tf.sign(Y_pred_valid-0.5)+1)
Y_pred_test_set = tf.sign(tf.sign(Y_pred_test-0.5)+1)
accuracy_train = tf.contrib.metrics.accuracy(tf.to_int32(Y_pred_train_set), tf.to_int32(trainTarget))
accuracy_valid = tf.contrib.metrics.accuracy(tf.to_int32(Y_pred_valid_set), tf.to_int32(validTarget))
accuracy_test = tf.contrib.metrics.accuracy(tf.to_int32(Y_pred_test_set), tf.to_int32(testTarget))

#TRAINING FOR OPTIMAL LOGISTIC REGRESSION WITH ZERO WEIGHT DECAY
# set parameters for the model
learning_rate = 0.005           #this is the best learning rate yielding the optimal cross entropy and accuracy results
batch_size = 500
n_batches = math.floor(trainData.shape[0]/batch_size)
n_epochs = math.floor(5000/n_batches)

# create placeholders for features and labels 
X = tf.placeholder(tf.float64, [None, 784])
Y = tf.placeholder(tf.float64, [None, 1])

# create weights and bias variables for training
w = tf.Variable(tf.zeros([784, 1], dtype=tf.float64), name = "weights")
b = tf.Variable(tf.zeros([1], dtype=tf.float64), name = "bias")

# sigmoid input with logistic regression
logits = tf.matmul(X,w) + b

# computing cross entropy loss from logits
loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = Y, logits = logits))

# computing accuracy for given features and labels
Y_pred = tf.sigmoid(logits)
accuracy = tf.contrib.metrics.accuracy(tf.to_int32(tf.round(Y_pred)), tf.to_int32(Y))

# define Cost and Accuracy outputs per epoch for LOGISTIC REGRESSION
Epoch = list(range(1, n_epochs+1))
Cost_train = []
Accuracy_train = []
Cost_valid = []
Accuracy_valid = []
Cost_test = []
Accuracy_test = []

# define adam optimizer
optimizer = tf.train.AdamOptimizer(learning_rate, name = "Adam").minimize(loss)

with tf.Session() as sess:
    # initialize the necessary variables: w (weights) and b (bias)
    sess.run(tf.global_variables_initializer())
    
    start_time = time.process_time()
    # apply SGD optimizer over the minibatches for each epoch pass
    for i in range(n_epochs):
        for j in range(n_batches):
            X_batch, Y_batch = trainData[j*batch_size:(j+1)*batch_size], trainTarget[j*batch_size:(j+1)*batch_size]
            _, loss_batch = sess.run([optimizer, loss], feed_dict = {X: X_batch, Y: Y_batch})
    
        # compute Cost and Accuracy for given epoch pass on test set
        loss_train, accuracy_train = sess.run([loss, accuracy], feed_dict = {X: trainData, Y: trainTarget})
        loss_valid, accuracy_valid = sess.run([loss, accuracy], feed_dict = {X: validData, Y: validTarget})
        loss_test, accuracy_test = sess.run([loss, accuracy], feed_dict = {X: testData, Y: testTarget})

        # plotted with epoch values
        Cost_train.append(loss_train)
        Cost_valid.append(loss_valid)
        Cost_test.append(loss_test)
        Accuracy_train.append(accuracy_train)
        Accuracy_valid.append(accuracy_valid)
        Accuracy_test.append(accuracy_test)
    end_time = time.process_time()
    
    print("\n" +
          "Optimizer = " + optimizer.name +
          "\nLrate = " + str(learning_rate) +
          "\nTrainAcc = " + str(Accuracy_train[-1]) +
          "\nValidAcc = " + str(Accuracy_valid[-1]) +
          "\nTestAcc = " + str(Accuracy_test[-1]) +
          "\nTime elapsed = " + str(end_time - start_time))
    
    print("\n" +
          "Normal Equation Accuracies" +
          "\nTrain Accuracy = " + str(accuracy_train) + 
          "\nValid Accuracy = " + str(accuracy_valid) + 
          "\nTest Accuracy = " + str(accuracy_test))
    
    #Plot Cost and Accuracy per Epoch for Training and Validation Sets
    plt.figure(1)
    plt.plot(Epoch, Cost_train, label='Lrate='+str(learning_rate))
    plt.title('Train Cross Entropy Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss Value')
    plt.legend(ncol=1, loc='upper right')
    plt.grid(True)
    
    plt.figure(2)
    plt.subplot(131)
    plt.plot(Epoch, Accuracy_train, label='Lrate='+str(learning_rate))
    plt.title('Train Classification Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Percentage Accuracy')
    plt.legend(ncol=1, loc='lower right')
    plt.grid(True)
    
    plt.subplot(132)
    plt.plot(Epoch, Accuracy_valid, label='Lrate='+str(learning_rate))
    plt.title('Valid Classification Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Percentage Accuracy')
    plt.legend(ncol=1, loc='lower right')
    plt.grid(True)    
    
    plt.subplot(133)
    plt.plot(Epoch, Accuracy_test, label='Lrate='+str(learning_rate))
    plt.title('Test Classification Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Percentage Accuracy')
    plt.legend(ncol=1, loc='lower right')
    plt.grid(True)
plt.show()