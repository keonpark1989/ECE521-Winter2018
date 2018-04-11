import time
import math
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def data_segmentation(data_path, target_path, task):
    # task = 0 >> select the name ID targets for face recognition task
    # task = 1 >> select the gender ID targets for gender recognition task 
    data = np.load(data_path)/255
    data = np.reshape(data, [-1, 32*32])
    target = np.load(target_path)
    np.random.seed(45689)
    rnd_idx = np.arange(np.shape(data)[0])
    np.random.shuffle(rnd_idx)
    trBatch = int(0.8*len(rnd_idx))  
    validBatch = int(0.1*len(rnd_idx))
    trainData, validData, testData = data[rnd_idx[1:trBatch],:], \
                                       data[rnd_idx[trBatch+1:trBatch + validBatch],:],\
                                       data[rnd_idx[trBatch + validBatch+1:-1],:]
    trainTarget, validTarget, testTarget = target[rnd_idx[1:trBatch], task], \
                                  target[rnd_idx[trBatch+1:trBatch + validBatch], task],\
                                  target[rnd_idx[trBatch + validBatch + 1:-1], task]
    return trainData, validData, testData, trainTarget, validTarget, testTarget

(trainData, validData, testData, trainTarget, validTarget, testTarget) = data_segmentation('data.npy','target.npy',0)



# resize the input image format
trainData = trainData.reshape([trainData.shape[0], 32*32])
validData = validData.reshape([validData.shape[0], 32*32])
testData = testData.reshape([testData.shape[0], 32*32])

# one-hot encode y labels
trainTarget_onehot = np.zeros((trainTarget.size, 6))
trainTarget_onehot[np.arange(trainTarget.size), trainTarget] = 1
trainTarget_onehot

validTarget_onehot = np.zeros((validTarget.size, 6))
validTarget_onehot[np.arange(validTarget.size), validTarget] = 1
validTarget_onehot

testTarget_onehot = np.zeros((testTarget.size, 6))
testTarget_onehot[np.arange(testTarget.size), testTarget] = 1
testTarget_onehot


# Model parameters
learning_rate = [0.005,0.001,0.0001]
decay_coef = [0.,0.001,0.01,1]
batch_size = 300
n_iterations = 5000;
trainData_size = trainData.shape[0]
trainTarget_size = trainTarget_onehot.shape[0]
n_class =6

# Input data
X = tf.placeholder(tf.float64, [None, 32*32])
Y = tf.placeholder(tf.float64, [None, n_class])

# Trainable parameters
w = tf.Variable(tf.zeros([32*32, n_class], dtype=tf.float64), name = "weights")
b = tf.Variable(tf.zeros([n_class], dtype=tf.float64), name = "bias")


Y_pred = tf.add(tf.matmul(X, w), b)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=Y_pred)
N = tf.cast(tf.shape(X)[0], dtype=tf.float64)
unregulated_loss = tf.reduce_sum(cross_entropy) / N


# Test model
Y_pred_acc = tf.nn.softmax(Y_pred)
pred_correct = tf.equal(tf.argmax(Y_pred_acc, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(pred_correct, tf.float32))
        
n_batches = math.ceil(trainData_size/batch_size)
n_epochs = math.floor(n_iterations/n_batches)


Epoch = list(range(1, n_epochs+1))

for k in range(len(learning_rate)):
    
    for d in range(len(decay_coef)):
                   
            
        training_loss_value_set = []
        validation_loss_value_set = []
        test_loss_value_set = []
    
        training_accuracy_set = []
        validation_accuracy_set = []
        test_accuracy_set = []  
        
        loss = unregulated_loss + (decay_coef[d] / 2) * tf.reduce_sum(tf.square(w))
        optimizer = tf.train.AdamOptimizer(learning_rate[k]).minimize(loss)

    
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())


            start_time = time.process_time()
            # Train
            for i in range(n_epochs):
                for j in range(n_batches):
                    if j == n_batches-1:
                        X_batch= np.concatenate((trainData[j*batch_size:trainData.shape[0]], trainData[0:(j+1)*batch_size - trainData.shape[0]]), axis=0)
                        Y_batch= np.concatenate((trainTarget_onehot[j*batch_size:trainTarget_onehot.shape[0]], trainTarget_onehot[0:(j+1)*batch_size - trainTarget_onehot.shape[0]]), axis=0)
                        _, loss_value_batch, Y_pred_batch = sess.run([optimizer, loss, Y_pred], feed_dict = {X: X_batch, Y: Y_batch})

                    else:    
                        X_batch, Y_batch = trainData[j*batch_size:(j+1)*batch_size], trainTarget_onehot[j*batch_size:(j+1)*batch_size]
                        _, loss_value_batch, Y_pred_batch = sess.run([optimizer, loss, Y_pred], feed_dict = {X: X_batch, Y: Y_batch})

                # Get loss
                training_loss_value = sess.run([loss], feed_dict = {X: trainData, Y: trainTarget_onehot})
                valid_loss_value = sess.run([loss], feed_dict = {X: validData, Y: validTarget_onehot})
                training_loss_value_set.append(training_loss_value)
                validation_loss_value_set.append(valid_loss_value)

                # Get accuracy
                training_accuracy = sess.run(accuracy, feed_dict = {X: trainData, Y: trainTarget_onehot})
                valid_accuracy = sess.run(accuracy, feed_dict = {X: validData, Y: validTarget_onehot})
                training_accuracy_set.append(training_accuracy)
                validation_accuracy_set.append(valid_accuracy)
            end_time = time.process_time()

            print("Lrate=" + str(learning_rate[k]) + 
                  ", Decay Coefficient= "+str(decay_coef[d])+
                  ", TrainingAcc=" + str(training_accuracy) +
                  ", TestAcc=" + str(valid_accuracy) +
                  ", Time elapsed=" + str(end_time - start_time) +
                 ", Train Loss= " + str(training_loss_value_set[-1]) +
                 ", Test Loss= "+ str(validation_loss_value_set[-1]))

           
        plt.figure(figsize = (10,10))
        plt.subplot(211)
        plt.plot(Epoch, training_loss_value_set, '-', label='Training')
        plt.plot(Epoch, validation_loss_value_set, '-', label='Validation')
        plt.title('Cross Entropy Loss for Learning rate= {0} & Decay coefficient= {1}'.format(learning_rate[k],decay_coef[d]))
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(ncol=1, loc='upper right')
        plt.grid(True)

        plt.subplot(212)
        plt.plot(Epoch, training_accuracy_set, '-', label='Training')
        plt.plot(Epoch, validation_accuracy_set, '-', label='Validation')
        plt.title('Accuracy for Learning rate= {0} & Decay coefficient= {1}'.format(learning_rate[k],decay_coef[d]))
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend(ncol=1, loc='lower right')
        plt.grid(True)
        plt.show()