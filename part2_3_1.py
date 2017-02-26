import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

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

        np.random.seed(521)
        randIndx = np.arange(len(Data))
        np.random.shuffle(randIndx)
        Data = Data[randIndx]/255.
        Target = Target[randIndx]
        trainData, trainTarget = Data[:15000], Target[:15000]
        validData, validTarget = Data[15000:16000], Target[15000:16000]
        testData, testTarget = Data[16000:], Target[16000:]

        dataInput = tf.placeholder(tf.float32)
        targetInput = tf.placeholder(tf.float32)

        # M is the number of training examples, N is the number of features
        M = trainData.shape[0]
        N = trainData.shape[1]

        learning_rate = 0.01
        weight_decay = 3e-4
        batch_size = 500

        hidden_units = 100

        z1, flattened_w1 = hidden_layer(tf.transpose(dataInput), N, hidden_units)
        z2, flattened_w2 = hidden_layer(tf.nn.relu(z1), hidden_units, 10)

        stacked_weights = [flattened_w1, flattened_w2]

        error = cross_entropy(M, weight_decay, stacked_weights, z2, targetInput)
        acc = accuracy(z2, targetInput)

        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(error)

        sess = tf.InteractiveSession()
        init = tf.global_variables_initializer()
        sess.run(init)
        sess.graph.finalize()

        trainingLosses = []
        trainingAccuracies = []
        testingLosses = []
        testingAccuracies = []
        validationLosses = []
        validationAccuracies = []
        for j in range(100):
            print("Epoch " + str(j))
            for i in range(0, M, batch_size):
                # Perform the update
                sess.run([optimizer], feed_dict={
                            dataInput: trainData[i:i+batch_size-1],
                            targetInput: trainTarget[i:i+batch_size-1]
                         })
            err, accuracy = sess.run([error, acc], feed_dict={
                        dataInput: trainData,
                        targetInput: trainTarget
                     })
            trainingLosses.append(err)
            trainingAccuracies.append(accuracy)

            err, accuracy = sess.run([error, acc], feed_dict={
                        dataInput: testData,
                        targetInput: testTarget
                     })
            testingLosses.append(err)
            testingAccuracies.append(accuracy)

            err, accuracy = sess.run([error, acc], feed_dict={
                        dataInput: validData,
                        targetInput: validTarget
                     })
            validationLosses.append(err)
            validationAccuracies.append(accuracy)
        print(testAccuracies[-1])
        plt.figure(1)
        plt.plot(trainingLosses)
        plt.title('Neural Net Training Loss vs. Number of Epochs')
        plt.figure(2)
        plt.plot(testingLosses)
        plt.title('Neural Net Testing Loss vs. Number of Epochs')
        plt.figure(3)
        plt.plot(validationLosses)
        plt.title('Neural Net Validation Loss vs. Number of Epochs')
        plt.figure(4)
        plt.plot(trainingAccuracies)
        plt.title('Neural Net Training Classification Accuracy vs. Number of Epochs')
        plt.figure(5)
        plt.plot(testingAccuracies)
        plt.title('Neural Net Testing Classification Accuracy vs. Number of Epochs')
        plt.figure(6)
        plt.plot(validationAccuracies)
        plt.title('Neural Net Validation Classification Accuracy vs. Number of Epochs')
        plt.show()
