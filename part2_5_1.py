import tensorflow as tf
import numpy as np
import time
import sys

def hidden_layer(h, num_inputs, num_units):
    # h is the input from the previous layer
    # num_inputs is the dimension of the vector h
    # num_units is the number of units in this layer
    N = num_inputs
    M = num_units

    # Xavier initialization
    W = tf.Variable(tf.random_normal(shape=[M, N], stddev=(3.0 / tf.sqrt(tf.to_float(N + M)))))
    b = tf.Variable(0, dtype=tf.float32)

    return tf.matmul(W, h) + b, tf.reshape(W, [-1])

def cross_entropy(M, l, stacked_weights, prediction, target):
    # One-hot encode the target to a matrix of 10 classes x N examples
    target_one_hot = tf.transpose(tf.one_hot(tf.to_int32(target), 10, 1.0, 0.0, -1))
    loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=target_one_hot, dim=0)) / M
    # Sum up the squares of all elements in W (weight decay)
    weight_decay = 0
    for weights in stacked_weights:
        weight_decay = weight_decay +  l / 2 * tf.reduce_sum(tf.square(weights))

    return loss + weight_decay

def accuracy(prediction, target):
    # Take the mean of the number of correctly predicted targets => accuracy
    return tf.reduce_mean(tf.to_float(tf.equal(tf.argmax(prediction, 0), tf.to_int64(target))), 0)

if __name__ == '__main__':
    # Load the two-class notMNIST dataset
    with np.load("notMNIST.npz") as data:
        Data, Target = data ["images"], data["labels"]

        # Flatten the images
        Data = np.reshape(Data, [-1, Data.shape[-1] ** 2])

        numpy_seed = int(time.time())
        np.random.seed(numpy_seed)

        tensorflow_seed = np.random.randint(sys.maxint)
        tf.set_random_seed(tensorflow_seed)

        randIndx = np.arange(len(Data))
        np.random.shuffle(randIndx)
        Data = Data[randIndx]/255.
        Target = Target[randIndx]
        trainData, trainTarget = Data[:15000], Target[:15000]
        validData, validTarget = Data[15000:16000], Target[15000:16000]
        testData, testTarget = Data[16000:], Target[16000:]

        # M is the number of training examples, N is the number of features
        M = trainData.shape[0]
        N = trainData.shape[1]

        batch_size = 500

        print("Numpy seed: " + str(numpy_seed))
        print("Tensorflow seed: " + str(tensorflow_seed))
        for x in range(5):
            print("\n\n\nStarting model " + str(x + 1))

            tf.reset_default_graph()
            dataInput = tf.placeholder(tf.float32)
            targetInput = tf.placeholder(tf.float32)
            keep_prob = tf.placeholder(tf.float32)

            min_learning_rate = np.exp(-7.5)
            max_learning_rate = np.exp(-4.5)

            min_weight_decay = np.exp(-9)
            max_weight_decay = np.exp(-6)

            learning_rate = (max_learning_rate - min_learning_rate) * np.random.random() + min_learning_rate
            weight_decay = (max_weight_decay - min_weight_decay) * np.random.random() + min_weight_decay
            use_dropout = np.random.random() > 0.5
            num_layers = np.random.randint(low=1, high=5)

            print("Learning rate: " + str(learning_rate))
            print("Weight decay: " + str(weight_decay))
            print("Use dropout: " + str(use_dropout))
            print("Number of layers: " + str(num_layers))

            z_prev = tf.transpose(dataInput)
            stacked_weights = []
            prev_hidden_units = N
            for i in range(num_layers):
                hidden_units = np.random.randint(low=100, high=500)
                print("Hidden units for layer " + str(i + 1) + ": " + str(hidden_units))
                z_prev, flattened_w = hidden_layer(z_prev, prev_hidden_units, hidden_units)
                z_prev = tf.nn.dropout(tf.nn.relu(z_prev), keep_prob)
                stacked_weights.append(flattened_w)
                prev_hidden_units = hidden_units

            # Output layer
            z2, flattened_w2 = hidden_layer(z_prev, prev_hidden_units, 10)

            stacked_weights.append(flattened_w2)

            error = cross_entropy(M, weight_decay, stacked_weights, z2, targetInput)
            acc = accuracy(z2, targetInput)

            optimizer = tf.train.AdamOptimizer(learning_rate).minimize(error)

            sess = tf.InteractiveSession()
            init = tf.global_variables_initializer()
            sess.run(init)
            #saver = tf.train.Saver()
            #sess.graph.finalize()

            trainingLosses = []
            trainingAccuracies = []
            testingLosses = []
            testingAccuracies = []
            validationLosses = []
            validationAccuracies = []

            for j in range(200):
                #print("Epoch " + str(j))
                for i in range(0, M, batch_size):
                    # Perform the update
                    sess.run([optimizer], feed_dict={
                                dataInput: trainData[i:i+batch_size-1],
                                targetInput: trainTarget[i:i+batch_size-1],
                                keep_prob: 0.5 if use_dropout else 1.0
                             })

            validationAccuracy = sess.run([acc], feed_dict={
                                dataInput: validData,
                                targetInput: validTarget,
                                keep_prob: 1.0
                             })

            testingAccuracy = sess.run([acc], feed_dict={
                                dataInput: testData,
                                targetInput: testTarget,
                                keep_prob: 1.0
                             })
            print("Validation accuracy: " + str(validationAccuracy))
            print("Testing accuracy: " + str(testingAccuracy))
            print("Finished model " + str(x + 1))

            sess.close()
