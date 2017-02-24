import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def cross_entropy(M, w, b, l, x, y):
    loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(tf.expand_dims(tf.reduce_sum(tf.reduce_sum(x * w, -1), -1), -1) + b, y), 0) / M
    weight_decay = l / 2 * tf.reduce_sum(tf.reduce_sum(tf.square(w), 0), 0)

    return loss + weight_decay

def accuracy(M, w, b, x, y):
    # Predict 1 if value is > 0.5, 0 otherwise
    prediction = tf.to_float(tf.greater(tf.reduce_sum(tf.reduce_sum(x * w, 1), 1) + b, 0.5))

    # 1 - abs(prediction - y)
    # Correct prediction => 1, incorrect prediction => 0
    return tf.reduce_mean(1 - tf.abs(prediction - tf.squeeze(y)), 0)

if __name__ == '__main__':
    # Load the two-class notMNIST dataset
    with np.load("notMNIST.npz") as data:
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

        dataInput = tf.placeholder(tf.float32)
        targetInput = tf.placeholder(tf.float32)

        # M is the number of training examples, N is the number of features
        M = trainData.shape[0]
        N = trainData.shape[1]

        w = tf.Variable(tf.zeros([N, N]), dtype=tf.float32)
        b = tf.Variable(0, dtype=tf.float32)
        learning_rate = 0.01
        weight_decay = 0.01
        batch_size = 500

        error = cross_entropy(M, w, b, weight_decay, dataInput, targetInput)
        acc = accuracy(M, w, b, dataInput, targetInput)

        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(error)

        sess = tf.InteractiveSession()
        init = tf.global_variables_initializer()
        sess.run(init)

        trainingLosses = []
        trainingAccuracies = []
        testingLosses = []
        testingAccuracies = []
        for j in range(100):
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
        print(testingAccuracies[-1])
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
