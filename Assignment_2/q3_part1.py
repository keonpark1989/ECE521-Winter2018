import time
import math
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Step 1: read in data from A2 provided files and reshape accordingly
def loadData():
    
    with np.load("notMNIST.npz") as data:
        Data, Target = data ["images"], data["labels"]
        np.random.seed(521)
        randIndx = np.arange(len(Data))
        np.random.shuffle(randIndx)
        Data = Data[randIndx]/255.
        Target = Target[randIndx]
        trainData, trainTarget = Data[:15000], Target[:15000]
        validData, validTarget = Data[15000:16000], Target[15000:16000]
        testData, testTarget = Data[16000:], Target[16000:]
        
    return trainData, trainTarget, validData, validTarget, testData, testTarget


(trainData, trainTarget, validData, validTarget, testData, testTarget) = loadData()

# resize the input image format
trainData = trainData.reshape([trainData.shape[0], 28*28])
validData = validData.reshape([validData.shape[0], 28*28])
testData = testData.reshape([testData.shape[0], 28*28])

# one-hot encode y labels
trainTarget_onehot = np.zeros((trainTarget.size, 10))
trainTarget_onehot[np.arange(trainTarget.size), trainTarget] = 1
trainTarget_onehot

validTarget_onehot = np.zeros((validTarget.size, 10))
validTarget_onehot[np.arange(validTarget.size), validTarget] = 1
validTarget_onehot

testTarget_onehot = np.zeros((testTarget.size, 10))
testTarget_onehot[np.arange(testTarget.size), testTarget] = 1
testTarget_onehot

# Model parameters
learning_rate = [0.005,0.001,0.0001]
decay_coef = 0.01
batch_size = 500
n_iterations = 5000;
trainData_size = trainData.shape[0]
n_class = 10

# Input data
X = tf.placeholder(tf.float64, [None, 784])
Y = tf.placeholder(tf.float64, [None, n_class])

# Trainable parameters
w = tf.Variable(tf.zeros([784, n_class], dtype=tf.float64), name = "weights")
b = tf.Variable(tf.zeros([n_class], dtype=tf.float64), name = "bias")

Y_pred = tf.add(tf.matmul(X, w), b)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=Y_pred)
unregulated_loss = tf.reduce_mean(cross_entropy)
loss = unregulated_loss + (decay_coef / 2) * tf.reduce_sum(tf.square(w))

Y_pred_acc = tf.nn.softmax(Y_pred)
pred_correct = tf.equal(tf.argmax(Y_pred_acc, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(pred_correct, tf.float32))

n_batches = math.ceil(trainData_size/batch_size)
n_epochs = math.floor(n_iterations/n_batches)

Epoch = list(range(1, n_epochs+1))

for k in range(len(learning_rate)):
    training_loss_value_set = []
    validation_loss_value_set = []
    test_loss_value_set = []
    
    training_accuracy_set = []
    validation_accuracy_set = []
    test_accuracy_set = []

    optimizer = tf.train.AdamOptimizer(learning_rate[k]).minimize(loss)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        start_time = time.process_time()
        # Train
        for i in range(n_epochs):
            for j in range(n_batches):
                X_batch, Y_batch = trainData[j*batch_size:(j+1)*batch_size], trainTarget_onehot[j*batch_size:(j+1)*batch_size]
                _, loss_value_batch, Y_pred_batch = sess.run([optimizer, loss, Y_pred], feed_dict = {X: X_batch, Y: Y_batch})

            # Get loss
            training_loss_value = sess.run([loss], feed_dict = {X: trainData, Y: trainTarget_onehot})
            valid_loss_value = sess.run([loss], feed_dict = {X: testData, Y: testTarget_onehot})
            training_loss_value_set.append(training_loss_value)
            validation_loss_value_set.append(valid_loss_value)

            # Get accuracy
            training_accuracy = sess.run(accuracy, feed_dict = {X: trainData, Y: trainTarget_onehot})
            valid_accuracy = sess.run(accuracy, feed_dict = {X: testData, Y: testTarget_onehot})
            training_accuracy_set.append(training_accuracy)
            validation_accuracy_set.append(valid_accuracy)
        end_time = time.process_time()

        print("Lrate=" + str(learning_rate[k]) +
              ", TrainingAcc=" + str(training_accuracy) +
              ", TestAcc=" + str(valid_accuracy) +
              ", Time elapsed=" + str(end_time - start_time) +
             ", Train Loss= " + str(training_loss_value_set[-1]) +
             ", Test Loss= "+ str(validation_loss_value_set[-1]))


        plt.figure(figsize = (10,10))
        plt.subplot(211)
        plt.plot(Epoch, training_loss_value_set, '-', label='Training')
        plt.plot(Epoch, validation_loss_value_set, '-', label='Test')
        plt.title('Cross Entropy Loss for Learning rate= {0}'.format(learning_rate[k]))
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(ncol=1, loc='upper right')
        plt.grid(True)


        plt.subplot(212)
        plt.plot(Epoch, training_accuracy_set, '-', label='Training')
        plt.plot(Epoch, validation_accuracy_set, '-', label='Test')
        plt.title('Accuracy for Learning rate= {0}'.format(learning_rate[k]))
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend(ncol=1, loc='lower right')
        plt.grid(True)
        plt.show()
