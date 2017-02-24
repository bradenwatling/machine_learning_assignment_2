import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def MSE(M, w, b, l, x, y):
    loss = tf.reduce_sum(tf.square(tf.reduce_sum(w * x, 1) + b - tf.squeeze(y)), 0) / (2 * M)
    weight_decay = l / 2 * tf.reduce_sum(tf.reduce_sum(tf.square(w), 0), 0)

    return loss + weight_decay

if __name__ == '__main__':
    with np.load ("tinymnist.npz") as data :
        trainData, trainTarget = data ["x"], data["y"]
        validData, validTarget = data ["x_valid"], data ["y_valid"]
        testData, testTarget = data ["x_test"], data ["y_test"]

        #trainData[:,5] *= 2
        #trainData[:,0] += 3

        validationData = tf.placeholder(tf.float32)
        validationTarget = tf.placeholder(tf.float32)
        trainingData = tf.placeholder(tf.float32)
        trainingTarget = tf.placeholder(tf.float32)
        testingData = tf.placeholder(tf.float32)
        testingTarget = tf.placeholder(tf.float32)

        # M is the number of training examples, N is the number of features
        M = trainData.shape[0]
        N = trainData.shape[1]

        w = tf.Variable(tf.zeros([1, N]), dtype=tf.float32)
        b = tf.Variable(0, dtype=tf.float32)
        learning_rate = 1.0
        weight_decay = 1.
        batch_size = 10

        mse = MSE(M, w, b, weight_decay, trainingData, trainingTarget)

        optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(mse)

        sess = tf.InteractiveSession()
        init = tf.global_variables_initializer()
        sess.run(init)

        losses = []
        for j in range(10):
            for i in range(0, trainData.shape[0], batch_size):
                losses.append(sess.run([mse, optimizer], feed_dict={
                            trainingData: trainData[i:i+batch_size-1],
                            trainingTarget: trainTarget[i:i+batch_size-1]
                         }))
        plt.plot(losses)
        plt.title('SGD Linear Regression Loss Function B=' + str(batch_size) + ', lambda=' + str(weight_decay) + ', eta=' + str(learning_rate))
        plt.show()
