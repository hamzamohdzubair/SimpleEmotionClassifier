import random
import numpy as np
import tensorflow as tf


def split_data(data, test_percent=0.2):
    """ Splits input data into training and testing data.
    :param data: A list to be split
    :type data: A list
    :param test_percent: The percentage for testing data. Default is 0.2.
    :type test_percent: A floating point value between 0 and 1.
    :return: Two lists, training data and testing data.
    :rtype: Two lists.
    """
    test_length = int(round(test_percent * len(data)))
    shuffled = data[:]
    random.shuffle(shuffled)
    training_data = shuffled[test_length:]
    testing_data = shuffled[:test_length]
    return training_data, testing_data


class EmotionClassifier:

    def __init__(self, num_classes, save_path=''):
        """ Constructor for EmotionClassifier that builds placeholders and the learning model.
        :param num_classes: The number of different classifications.
        :type num_classes: int
        :param save_path: A file path for the session variables to be saved. If not set the session will not be saved.
        :type save_path: A file path.
        """
        self.x = tf.placeholder("float", [None, 134])
        self.y = tf.placeholder("float", [None, num_classes])
        self.model = self.build_model(num_classes)
        self.save_path = save_path

    def build_model(self, num_classes):
        """ Builds the Neural model for the classifier.
        :param num_classes: The number of different classifications.
        :type num_classes: int
        :return: The Neural model for the system.
        :rtype: A TensorFlow model.
        """
        weights = {
            'h1': tf.Variable(tf.random_normal([134, 256])),
            'h2': tf.Variable(tf.random_normal([256, 512])),
            'h3': tf.Variable(tf.random_normal([512, 256])),
            'out': tf.Variable(tf.random_normal([256, num_classes]))
        }
        biases = {
            'b1': tf.Variable(tf.random_normal([256])),
            'b2': tf.Variable(tf.random_normal([512])),
            'b3': tf.Variable(tf.random_normal([256])),
            'out': tf.Variable(tf.random_normal([num_classes]))
        }

        layer1 = tf.nn.relu(tf.add(tf.matmul(self.x, weights['h1']), biases['b1']))
        layer2 = tf.nn.relu(tf.add(tf.matmul(layer1, weights['h2']), biases['b2']))
        layer3 = tf.nn.relu(tf.add(tf.matmul(layer2, weights['h3']), biases['b3']))
        return tf.matmul(layer3, weights['out']) + biases['out']

    def train(self, training_data, testing_data, epochs=50000):
        """

        :param training_data: A list of tuples used for training the classifier.
        :type training_data: A list of tuples each containing a list of landmarks and a list of classifications.
        :param testing_data: A list of tuples used for testing the classifier.
        :type testing_data: A list of tuples each containing a list of landmarks and a list of classifications.
        :param epochs: The number of cycles to train the classifier for. Default is 50000.
        :type epochs: int
        """
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.model, self.y))
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)
        init, saver = tf.initialize_all_variables(), tf.train.Saver()

        with tf.Session() as sess:
            sess.run(init)
            correct_prediction = tf.equal(tf.argmax(self.model, 1), tf.argmax(self.y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

            for epoch in range(epochs):
                batch_x, batch_y = [m[0] for m in training_data], [n[1] for n in training_data]
                _, avg_cost = sess.run([optimizer, cost], feed_dict={self.x: batch_x, self.y: batch_y})
                if epoch % 5000 == 0:
                    print "Epoch", '%04d' % (epoch), "cost = ", "{:.9f}".format(avg_cost)

            print "Optimization Finished!"
            saver.save(sess, self.save_path) if self.save_path != '' else ''
            print "Accuracy:", accuracy.eval({self.x: [m[0] for m in testing_data],
                                              self.y: [n[1] for n in testing_data]})

    def classify(self, data):
        init, saver = tf.initialize_all_variables(), tf.train.Saver()
        with tf.Session() as sess:
            sess.run(init)
            saver.restore(sess, self.save_path)
            classification = np.asarray(sess.run(self.model, feed_dict={self.x: data}))
            return np.unravel_index(classification.argmax(), classification.shape)
