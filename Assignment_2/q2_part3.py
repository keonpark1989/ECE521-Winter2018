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

# Model parameters
learning_rate = [0.001]
decay_coef = 0
batch_size = 500
n_iterations = 5000
trainData_size = trainData.shape[0]

# Input data
X = tf.placeholder(tf.float64, [None, 784])
Y = tf.placeholder(tf.float64, [None, 1])

# Trainable parameters
w = tf.Variable(tf.zeros([784, 1], dtype=tf.float64), name = "weights")
b = tf.Variable(tf.zeros([1], dtype=tf.float64), name = "bias")

Y_logits = tf.add(tf.matmul(X, w), b)
Y_pred = tf.sigmoid(tf.add(tf.matmul(X, w), b))

mse_accuracy = tf.contrib.metrics.accuracy(tf.to_int32(Y), tf.to_int32(tf.round(Y_logits)))
cee_accuracy = tf.contrib.metrics.accuracy(tf.to_int32(Y), tf.to_int32(tf.round(Y_pred)))

n_batches = math.ceil(trainData_size/batch_size)
#n_batches = math.floor(3500/1500) + 1
n_epochs = math.floor(n_iterations/n_batches)
#n_epochs = math.floor(20000/3)

Epoch = list(range(1, n_epochs+1))

for n in range(len(learning_rate)):
    # Logistic Regression Loss Function
    cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=Y, logits=Y_logits)
    unregulated_loss = tf.reduce_mean(cross_entropy)
    logistic_regression_loss = unregulated_loss + (decay_coef / 2) * tf.reduce_sum(tf.square(w))
    # Linear Regression Loss Function
    linear_regression_loss = tf.reduce_mean(tf.square(Y_logits - Y)) / 2
    loss = [logistic_regression_loss, linear_regression_loss]
    accuracy = [cee_accuracy, mse_accuracy]
    for k in range(len(loss)):
        optimizer = tf.train.AdamOptimizer(learning_rate[n], name="Adam").minimize(loss[k])
        training_loss_value_set = []
        validation_loss_value_set = []
        training_accuracy_set = []
        validation_accuracy_set = []

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())

            start_time = time.process_time()
            # Train
            for i in range(n_epochs):
                for j in range(n_batches):
                    X_batch, Y_batch = trainData[j*batch_size:(j+1)*batch_size], trainTarget[j*batch_size:(j+1)*batch_size]
                    _, loss_value_batch, Y_pred_batch = sess.run([optimizer, loss[k], Y_pred], feed_dict = {X: X_batch, Y: Y_batch})

                # Get loss
                training_loss_value = sess.run([loss[k]], feed_dict = {X: trainData, Y: trainTarget})
                validation_loss_value = sess.run([loss[k]], feed_dict = {X: validData, Y: validTarget})
                training_loss_value_set.append(training_loss_value)
                validation_loss_value_set.append(validation_loss_value)

                # Get accuracy
                training_accuracy = sess.run(accuracy[k], feed_dict = {X: trainData, Y: trainTarget})
                validation_accuracy = sess.run(accuracy[k], feed_dict = {X: validData, Y: validTarget})
                training_accuracy_set.append(training_accuracy)
                validation_accuracy_set.append(validation_accuracy)
            end_time = time.process_time()

        loss_string = str("CEE" if k == 0 else "MSE")
        print("Loss=" + loss_string +
              ", Lrate=" + str(learning_rate[n]) +
              ", TrainingAcc=" + str(training_accuracy) +
              ", ValidationAcc=" + str(validation_accuracy) +
              ", Time elapsed=" + str(end_time - start_time))

        # Step 10: plotting the results
        plt.plot(Epoch, training_accuracy_set, '-', label='Training, loss=' + loss_string)
        #plt.plot(Epoch, validation_loss_value_set, '-', label=' Validation, loss=' + loss_string)

    plt.title(optimizer.name + ' Optimizer')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(ncol=1, loc='right')
    plt.grid(True)
    plt.show()
