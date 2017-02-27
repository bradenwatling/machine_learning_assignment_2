import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def cross_entropy(M, w, b, l, x, y):
    # Add a dimension so that the shape of x is [num_examples, 1, 28, 28]
    x = tf.expand_dims(x, 1)
    prediction = tf.reduce_sum(tf.reduce_sum(x * w, -1), -1) + b
    # One-hot encode the target to a matrix of 10 classes x N examples
    target_one_hot = tf.one_hot(tf.to_int32(y), 10, 1.0, 0.0, -1)
    loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=target_one_hot, dim=-1)) / M
    weight_decay = l / 2 * tf.reduce_sum(tf.square(w))

    return loss + weight_decay

def accuracy(w, b, x, y):
    # Add a dimension so that the shape of x is [num_examples, 1, 28, 28]
    x = tf.expand_dims(x, 1)
    prediction = tf.nn.softmax(tf.reduce_sum(tf.reduce_sum(x * w, -1), -1) + b)

    # Take the mean of the number of correctly predicted targets => accuracy
    return tf.reduce_mean(tf.to_float(tf.equal(tf.argmax(prediction, -1), tf.to_int64(y))), 0)

if __name__ == '__main__':
    # Load the two-class notMNIST dataset
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

        dataInput = tf.placeholder(tf.float32)
        targetInput = tf.placeholder(tf.float32)

        # M is the number of training examples, N is the number of features
        M = trainData.shape[0]
        N = trainData.shape[1]

        w = tf.Variable(tf.zeros([10, N, N]), dtype=tf.float32)
        b = tf.Variable(0, dtype=tf.float32)
        learning_rate = 0.01
        weight_decay = 0.01
        batch_size = 500

        error = cross_entropy(M, w, b, weight_decay, dataInput, targetInput)
        acc = accuracy(w, b, dataInput, targetInput)

        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(error)

        sess = tf.InteractiveSession()
        init = tf.global_variables_initializer()
        sess.run(init)

        trainingLosses = []
        trainingAccuracies = []
        testingLosses = []
        testingAccuracies = []
        for j in range(100):
            print("Epoch " + str(j))
            for i in range(0, M, batch_size):
                # Perform the update
                _, err, accuracy = sess.run([optimizer, error, acc], feed_dict={
                            dataInput: trainData[i:i+batch_size-1],
                            targetInput: trainTarget[i:i+batch_size-1]
                         })
                trainingLosses.append(err)
                trainingAccuracies.append(accuracy)
                err, accuracy = sess.run([error, acc], feed_dict={
                            dataInput: testData,
                            targetInput: testTarget
                         })
                testingLosses.append(err)
                testingAccuracies.append(accuracy)
        print("Max testing accuracy: " + str(np.amax(testingAccuracies)))
        plt.figure(1)
        plt.plot(trainingLosses)
        plt.title('SGD Logistic Regression Training Loss vs. Number of Updates')
        plt.figure(2)
        plt.plot(testingLosses)
        plt.title('SGD Logistic Regression Testing Loss vs. Number of Updates')
        plt.figure(3)
        plt.plot(trainingAccuracies)
        plt.title('SGD Logistic Regression Training Classification Accuracy vs. Number of Updates')
        plt.figure(4)
        plt.plot(testingAccuracies)
        plt.title('SGD Logistic Regression Testing Classification Accuracy vs. Number of Updates')
        plt.show()
