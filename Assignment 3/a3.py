#!/usr/bin/env python3
import math
import matplotlib.pyplot as plt
import numpy as np
import sys
import tensorflow as tf

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

# Get input data
(trainData, trainTarget, validData, validTarget, testData, testTarget) = loadData()
# resize the input data format
trainData = trainData.reshape([trainData.shape[0], 28*28])
validData = validData.reshape([validData.shape[0], 28*28])
testData = testData.reshape([testData.shape[0], 28*28])

# GLOBAL VARIABLES
num_classes = 10

# Part 1: layer-wise building block
def calculate_s(data, num_hidden_units):
    # data is N x d
    # w is d x M, where d = dim(input vector), M = number of hidden units
    w = tf.get_variable("weights", shape = [data.shape[1], num_hidden_units],
                        initializer = tf.contrib.layers.xavier_initializer())
    # b is 1 x M, where M = number of hidden units
    b = tf.get_variable("bias", shape=[1, num_hidden_units],
                        initializer = tf.zeros_initializer)
    # s is N x M
    s = tf.add(tf.matmul(data, w), b)
    return s

# Part 2: Learning
def create_nn_model(x, config, do_dropout):
    for i in range(1, config.num_hidden_layers + 1):
        # x is N x num_hidden_units
        with tf.variable_scope("hidden_layer_" + str(i)):
            s = calculate_s(x, config.num_hidden_units)
        x = tf.nn.relu(s)
        x = tf.layers.dropout(x, rate = config.dropout_keep_prob, training=do_dropout)

    with tf.variable_scope("output_layer"):
        s = calculate_s(x, num_classes)

    # output is N x num_classes
    return s

def get_prediction(nn_out):
    return tf.cast(tf.argmax(tf.nn.softmax(nn_out), axis=1), dtype=tf.int32)

def classification_error(nn_out, target):
    acc = tf.cast(tf.equal(get_prediction(nn_out), target), tf.float32)
    return 1 - tf.reduce_mean(acc)

def cross_entropy_loss(nn_out, target, config):
    # Regularization value
    reg = 0
    for i in range(1, config.num_hidden_layers + 1):
        with tf.variable_scope("hidden_layer_" + str(i), reuse = True):
            w = tf.get_variable('weights')
            reg += (config.weight_decay) * tf.reduce_sum(tf.square(w))

    # Cross Entropy
    one_hot_target = tf.one_hot(target, num_classes, dtype=tf.float32)
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_target,
                                                            logits=nn_out)
    total_cross_entropy = tf.reduce_mean(cross_entropy)

    # Sum them together
    loss = total_cross_entropy + reg
    return loss

def get_random(upper, lower):
    assert(lower >= 0)
    assert(upper >= lower)
    return ((upper + 1) - lower) * np.random.random_sample() + lower

class ModelConfig:
    def __init__(self):
        self.n_epochs = 100
        self.batch_size = 500
        self.learning_rate = 0.0001
        self.weight_decay = 3e-4
        self.num_hidden_layers = 1
        self.num_hidden_units = 1000
        self.dropout_keep_prob = 1.0

def train_model(config):
    tf.reset_default_graph()
    n_batches = math.ceil(trainData.shape[0] / config.batch_size)
    X = tf.placeholder(tf.float32, [None, 784])
    Y = tf.placeholder(tf.int32, [None])
    do_dropout = tf.placeholder(tf.bool)
    nn_out = create_nn_model(X, config, do_dropout)
    loss = cross_entropy_loss(nn_out, Y, config)
    error = classification_error(nn_out, Y)
    optimizer = tf.train.AdamOptimizer(config.learning_rate, name="Adam").minimize(loss)

    training_loss_value_set = []
    validation_loss_value_set = []
    test_loss_value_set = []
    training_error_set = []
    validation_error_set = []
    test_error_set = []

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # Train
        for i in range(config.n_epochs):
            for j in range(n_batches):
                X_batch, Y_batch = trainData[j*config.batch_size:(j+1)*config.batch_size], trainTarget[j*config.batch_size:(j+1)*config.batch_size]
                # Train model
                _temp = sess.run(optimizer, feed_dict = {X: X_batch, Y: Y_batch, do_dropout: True})

            # Get loss
            training_loss_value = sess.run(loss, feed_dict = {X: trainData, Y: trainTarget, do_dropout: False})
            training_loss_value_set.append(training_loss_value)
            validation_loss_value = sess.run(loss, feed_dict = {X: validData, Y: validTarget, do_dropout: False})
            validation_loss_value_set.append(validation_loss_value)
            test_loss_value = sess.run(loss, feed_dict = {X: testData, Y: testTarget, do_dropout: False})
            test_loss_value_set.append(test_loss_value)

            # Get accuracy
            training_error = sess.run(error, feed_dict = {X: trainData, Y: trainTarget, do_dropout: False})
            training_error_set.append(training_error)
            validation_error = sess.run(error, feed_dict = {X: validData, Y: validTarget, do_dropout: False})
            validation_error_set.append(validation_error)
            test_error = sess.run(error, feed_dict = {X: testData, Y: testTarget, do_dropout: False})
            test_error_set.append(test_error)

    return (training_loss_value_set, validation_loss_value_set, test_loss_value_set,
            training_error_set, validation_error_set, test_error_set)

# Part 3: Resize the weight matrix into a grayscale image
def config_image_h1(wt):
    # configuring the first 100 hidden units as images for display
    wtMat_temp = wt[0:100,:]
    
    # find padding constant for each hidden unit image
    pad_index = tf.argmin(tf.reshape(wtMat_temp, [100*784]), axis = 0)
    pad_const = tf.reshape(wtMat_temp, [100*784])[pad_index]
    
    # adding in the padding: output is 100 x 30 x 30 tensor
    wtMat = tf.reshape(wtMat_temp, [100, 28, 28])
    wtMat = tf.pad(wtMat,[[0,0],[1,1],[1,1]],"CONSTANT",constant_values=pad_const)
    
    # reformat to acquire a 300 x 300 tensor for grayscale image
    wtMat = tf.unstack(wtMat, num = 100, axis = 0, name = 'unstack')
    wtMat = tf.concat(wtMat, 0)
    wtMat = tf.split(wtMat, 10, 0)
    wtMat = tf.concat(wtMat, 1)
    
    return wtMat


################################
##### Assignment Questions #####
################################
def part11_2_choosing_learning_rate():
    learning_rates = [0.005, 0.001, 0.0001]
    n_epochs = 100
    Epoch = list(range(1, n_epochs+1))
    for learning_rate in learning_rates:
        config = ModelConfig()
        config.n_epochs = n_epochs
        config.batch_size = 500
        config.learning_rate = learning_rate
        config.weight_decay = 3e-4
        config.num_hidden_layers = 1
        config.num_hidden_units = 1000
        config.dropout_keep_prob = 1.0

        model_output = train_model(config)
        training_loss_value_set = model_output[0]
        validation_loss_value_set = model_output[1]
        test_loss_value_set = model_output[2]
        training_error_set = model_output[3]
        validation_error_set = model_output[4]
        test_error_set = model_output[5]

        plt.plot(Epoch, validation_loss_value_set, '-',
                 label='Learning rate=' + str(learning_rate))
    plt.title("Cross Entropy Loss of Validation set")
    plt.xlabel('Epoch')
    plt.ylabel('Cross Entropy Loss')
    plt.legend(ncol=1, loc='upper right')
    plt.grid(True)
    plt.show()

def part11_2():
    config = ModelConfig()
    config.n_epochs = 100
    config.batch_size = 500
    config.learning_rate = 0.0001
    config.weight_decay = 3e-4
    config.num_hidden_layers = 1
    config.num_hidden_units = 1000
    config.dropout_keep_prob = 1.0

    model_output = train_model(config)
    training_loss_value_set = model_output[0]
    validation_loss_value_set = model_output[1]
    test_loss_value_set = model_output[2]
    training_error_set = model_output[3]
    validation_error_set = model_output[4]
    test_error_set = model_output[5]

    # Add data point to plot
    Epoch = list(range(1, config.n_epochs+1))
    plt.figure(figsize = (10,10))
    plt.subplot(211)
    plt.plot(Epoch, training_loss_value_set, '-', label='Training')
    plt.plot(Epoch, validation_loss_value_set, '-', label='Validation')
    plt.plot(Epoch, test_loss_value_set, '-', label='Test')
    plt.title("Cross Entropy Loss of Training, Validation, and Test data sets")
    plt.xlabel('Epoch')
    plt.ylabel('Cross Entropy Loss')
    plt.legend(ncol=1, loc='upper right')
    plt.grid(True)

    plt.subplot(212)
    plt.plot(Epoch, training_error_set, '-', label='Training')
    plt.plot(Epoch, validation_error_set, '-', label='Validation')
    plt.plot(Epoch, test_error_set, '-', label='Test')
    plt.title("Classification error of Training, Validation, and Test data sets")
    plt.xlabel('Epoch')
    plt.ylabel('Classification Error')
    plt.legend(ncol=1, loc='upper right')
    plt.grid(True)

    plt.show()

def part11_3():
    config = ModelConfig()
    config.n_epochs = 1000
    config.batch_size = 500
    config.learning_rate = 0.0001
    config.weight_decay = 3e-4
    config.num_hidden_layers = 1
    config.num_hidden_units = 1000
    config.dropout_keep_prob = 1.0

    model_output = train_model(config)
    training_loss_value_set = model_output[0]
    validation_loss_value_set = model_output[1]
    test_loss_value_set = model_output[2]
    training_error_set = model_output[3]
    validation_error_set = model_output[4]
    test_error_set = model_output[5]

    # Add data point to plot
    Epoch = list(range(1, config.n_epochs+1))
    plt.subplot(211)
    plt.plot(Epoch, training_loss_value_set, '-', label='Training')
    plt.plot(Epoch, validation_loss_value_set, '-', label='Validation')
    plt.plot(Epoch, test_loss_value_set, '-', label='Test')
    plt.title("Cross Entropy Loss of Training, Validation, and Test data sets")
    plt.xlabel('Epoch')
    plt.ylabel('Cross Entropy Loss')
    plt.legend(ncol=1, loc='upper right')
    plt.grid(True)

    plt.subplot(212)
    plt.plot(Epoch, training_error_set, '-', label='Training')
    plt.plot(Epoch, validation_error_set, '-', label='Validation')
    plt.plot(Epoch, test_error_set, '-', label='Test')
    plt.title("Classification Error of Training, Validation, and Test data sets")
    plt.xlabel('Epoch')
    plt.ylabel('Classification Error')
    plt.legend(ncol=1, loc='upper right')
    plt.grid(True)

    plt.show()

def part12_1():
    for num_hidden_units in [100, 500, 1000]:
        config = ModelConfig()
        config.n_epochs = 200
        config.batch_size = 500
        config.learning_rate = 0.0001
        config.weight_decay = 3e-4
        config.num_hidden_layers = 1
        config.num_hidden_units = num_hidden_units
        config.dropout_keep_prob = 1.0

        model_output = train_model(config)
        training_loss_value_set = model_output[0]
        validation_loss_value_set = model_output[1]
        test_loss_value_set = model_output[2]
        training_error_set = model_output[3]
        validation_error_set = model_output[4]
        test_error_set = model_output[5]

        print("##############################################3")
        print("Num Hidden Units=" + str(num_hidden_units))
        print("Cross Entropy Loss (Training):\t\t" + str(training_loss_value_set[-1]))
        print("Cross Entropy Loss (Validation):\t" + str(validation_loss_value_set[-1]))
        print("Cross Entropy Loss (Test):\t\t" + str(test_loss_value_set[-1]))
        print("Classification Error (Training):\t" + str(training_error_set[-1]))
        print("Classification Error (Validation):\t" + str(validation_error_set[-1]))
        print("Classification Error (Test):\t\t" + str(test_error_set[-1]))

def part12_2():
    config = ModelConfig()
    config.n_epochs = 100
    config.batch_size = 500
    config.learning_rate = 0.0001
    config.weight_decay = 3e-4
    config.num_hidden_layers = 2
    config.num_hidden_units = 500
    config.dropout_keep_prob = 1.0

    model_output = train_model(config)
    training_loss_value_set = model_output[0]
    validation_loss_value_set = model_output[1]
    test_loss_value_set = model_output[2]
    training_error_set = model_output[3]
    validation_error_set = model_output[4]
    test_error_set = model_output[5]

    # Add data point to plot
    Epoch = list(range(1, config.n_epochs+1))
    plt.figure(figsize = (15,10))
    plt.subplot(211)
    plt.plot(Epoch, training_loss_value_set, '-', label='Training')
    plt.plot(Epoch, validation_loss_value_set, '-', label='Validation')
#     plt.plot(Epoch, test_loss_value_set, '-', label='Test')
    plt.title("Cross Entropy Loss of Training and Validation data sets")
    plt.xlabel('Epoch')
    plt.ylabel('Cross Entropy Loss')
    plt.legend(ncol=1, loc='upper right')
    plt.grid(True)

    plt.subplot(212)
    plt.plot(Epoch, training_error_set, '-', label='Training')
    plt.plot(Epoch, validation_error_set, '-', label='Validation')
#     plt.plot(Epoch, test_error_set, '-', label='Test')
    plt.title("Classification error of Training and Validation data sets")
    plt.xlabel('Epoch')
    plt.ylabel('Classification Error')
    plt.legend(ncol=1, loc='upper right')
    plt.grid(True)

    plt.show()

def part13_1():
    config = ModelConfig()
    config.n_epochs = 200
    config.batch_size = 500
    config.learning_rate = 0.0001
    config.weight_decay = 3e-4
    config.num_hidden_layers = 1
    config.num_hidden_units = 1000
    config.dropout_keep_prob = 0.5

    model_output = train_model(config)
    training_loss_value_set = model_output[0]
    validation_loss_value_set = model_output[1]
    test_loss_value_set = model_output[2]
    training_error_set = model_output[3]
    validation_error_set = model_output[4]
    test_error_set = model_output[5]

    # Add data point to plot
    Epoch = list(range(1, config.n_epochs+1))
    plt.figure(figsize = (15,10))
    plt.subplot(211)
    plt.plot(Epoch, training_loss_value_set, '-', label='Training')
    plt.plot(Epoch, validation_loss_value_set, '-', label='Validation')
#     plt.plot(Epoch, test_loss_value_set, '-', label='Test')
    plt.title("Cross Entropy Loss of Training and Validation data sets")
    plt.xlabel('Epoch')
    plt.ylabel('Cross Entropy Loss')
    plt.legend(ncol=1, loc='upper right')
    plt.grid(True)

    plt.subplot(212)
    plt.plot(Epoch, training_error_set, '-', label='Training')
    plt.plot(Epoch, validation_error_set, '-', label='Validation')
#     plt.plot(Epoch, test_error_set, '-', label='Test')
    plt.title("Classification error of Training and Validation data sets")
    plt.xlabel('Epoch')
    plt.ylabel('Classification Error')
    plt.legend(ncol=1, loc='upper right')
    plt.grid(True)

    plt.show()

    print("Cross Entropy Loss (Training):\t\t" + str(training_loss_value_set[-1]))
    print("Cross Entropy Loss (Validation):\t" + str(validation_loss_value_set[-1]))
    print("Cross Entropy Loss (Test):\t\t" + str(test_loss_value_set[-1]))
    print("Classification Error (Training):\t" + str(training_error_set[-1]))
    print("Classification Error (Validation):\t" + str(validation_error_set[-1]))
    print("Classification Error (Test):\t\t" + str(test_error_set[-1]))

def part13_2():
    # define the directories for saved model states
    model_path_25 = "checkpoints/model_25.ckpt"
    model_path_50 = "checkpoints/model_50.ckpt"
    model_path_75 = "checkpoints/model_75.ckpt"
    model_path_100 = "checkpoints/model_100.ckpt"   

    n_epochs = 200
    batch_size = 500
    n_batches = math.ceil(trainData.shape[0] / batch_size)
    Epoch = list(range(1, n_epochs+1))

    learning_rate = 0.0001   #tune for optimal learning rate
    lambda_par = 3e-4

    config = ModelConfig()
    config.n_epochs = n_epochs
    config.batch_size = batch_size
    config.learning_rate = learning_rate
    config.weight_decay = lambda_par
    config.num_hidden_layers = 1
    config.num_hidden_units = 1000
    config.dropout_keep_prob = 1.0

    X = tf.placeholder(tf.float32, [None, 784])
    Y = tf.placeholder(tf.int32, [None])

    nn_out = create_nn_model(X, config, False)
    loss = cross_entropy_loss(nn_out, Y, config)
    error = classification_error(nn_out, Y)

    # define neural network optimizer using 'Adam' algorithm
    optimizer = tf.train.AdamOptimizer(learning_rate, name="Adam").minimize(loss)
    
    # storage for classification error and cross-entropy loss
    training_loss_value_set = []
    validation_loss_value_set = []
    test_loss_value_set = []
    training_error_set = []
    validation_error_set = []
    test_error_set = []
    
    # resize the weight matrix image with paddings
    with tf.variable_scope("hidden_layer_1", reuse = True):
        wt_h1 = tf.transpose(tf.get_variable('weights'))
    wt_padded = config_image_h1(wt_h1)

    # initialize the variables
    init = tf.global_variables_initializer()
    
    # 'saver' op to save and restore all the variables
    saver = tf.train.Saver()
    
    # Training Session: Update Weights and Save Checkpoints
    with tf.Session() as sess:
        sess.run(init)

        # Train
        for i in range(n_epochs):
            for j in range(n_batches):
                X_batch, Y_batch = trainData[j*batch_size:(j+1)*batch_size], trainTarget[j*batch_size:(j+1)*batch_size]
                _temp = sess.run([optimizer], feed_dict = {X: X_batch, Y: Y_batch})
            
            # Save Model State
            if i == int(0.25*n_epochs - 1):
                save_path = saver.save(sess, model_path_25)
                print("Model 25 percent trained saved in file: %s" % save_path)                
            elif i == int(0.50*n_epochs - 1):
                save_path = saver.save(sess, model_path_50)
                print("Model 50 percent trained saved in file: %s" % save_path)                
            elif i == int(0.75*n_epochs - 1):
                save_path = saver.save(sess, model_path_75)
                print("Model 75 percent trained saved in file: %s" % save_path)                
            elif i == n_epochs - 1:
                save_path = saver.save(sess, model_path_100)
                print("Model 100 percent trained saved in file: %s" % save_path)                
    # Open Model 25% Trained: Plot Weights of First Hidden Layer as Grayscale Image
    with tf.Session() as sess:
        sess.run(init)
        
        # restore model weights from previously saved model
        saver.restore(sess, model_path_25)
        
        # print padded matrix in grayscale
        wt_plot = sess.run(wt_padded)
        plt.figure()
        plt.imshow(wt_plot, cmap=plt.get_cmap('gray'))
        plt.title("Weight Grayscale Map: 25% Training")
        plt.axis('off')
        plt.show()
    
    # Open Model 100% Trained: Plot Weights of First Hidden Layer as Grayscale Image
    with tf.Session() as sess:
        sess.run(init)
        
        # restore model weights from previously saved model
        saver.restore(sess, model_path_100)
        
        # print padded matrix in grayscale
        wt_plot = sess.run(wt_padded)
        plt.figure()
        plt.imshow(wt_plot, cmap=plt.get_cmap('gray'))
        plt.title("Weight Grayscale Map: 100% Training")
        plt.axis('off')
        plt.show()    

def part14_1():
    # Seed the numpy and Tensorflow randomizers
    random_seed = 1000594270 + 1000548874 + 1004686253
    np.random.seed(random_seed)
    tf.set_random_seed(random_seed)

    config = ModelConfig()
    for i in range(5):
        # Randomize hyperparameters
        config.n_epochs = 100
        config.batch_size = 500
        config.learning_rate = math.exp(-1 * get_random(7.5, 4.5))
        config.num_hidden_layers = math.floor(get_random(5, 1))
        config.num_hidden_units = math.floor(get_random(500, 100))
        config.weight_decay = math.exp(-1 * get_random(9.0, 6.0))
        do_dropout = math.floor(get_random(1, 0))
        config.dropout_keep_prob = 0.5 if (do_dropout == 1) else 1.0

        print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        print("Run #" + str(i))
        print("Learning rate=" + str(config.learning_rate))
        print("Number of hidden layers=" + str(config.num_hidden_layers))
        print("Number of hidden units per layer=" + str(config.num_hidden_units))
        print("Regularization weight decay=" + str(config.weight_decay))
        print("Do dropout=" + ("yes" if (do_dropout == 1) else "no"))

        model_output = train_model(config)
        training_loss_value_set = model_output[0]
        validation_loss_value_set = model_output[1]
        test_loss_value_set = model_output[2]
        training_error_set = model_output[3]
        validation_error_set = model_output[4]
        test_error_set = model_output[5]

        print("Cross Entropy Loss (Training):\t\t" + str(training_loss_value_set[-1]))
        print("Cross Entropy Loss (Validation):\t" + str(validation_loss_value_set[-1]))
        print("Cross Entropy Loss (Test):\t\t" + str(test_loss_value_set[-1]))
        print("Classification Error (Training):\t" + str(training_error_set[-1]))
        print("Classification Error (Validation):\t" + str(validation_error_set[-1]))
        print("Classification Error (Test):\t\t" + str(test_error_set[-1]))

if __name__ == "__main__":
    #part11_2_choosing_learning_rate()
    #part11_2()
    #part11_3()
    #part12_1()
    #part12_2()
    part13_1()
    #part13_2()
    #part14_1()
