"""
    Ai Module
    Contains all the logic to create a playing AI
"""

import tensorflow as tf
from network import *
from summary import variable_summaries
from train import get_training_batch, get_test_set


class BernardAi:
    """
        Représente une AI capable de jouer au jeu
        Implémente un réseau de neurones TensorFlow
    """

    def __init__(self, name):
        self.name = name
        self.x = None
        self.W = None
        self.b = None
        self.y = None
        self.y_ = None

    def init(self, height, width):
        """
        Initialise le réseau de neurones
        """
        print("Initializing {} AI".format(self.name))

        self.image_vector_size = height * width

        # Image input
        self.x = tf.placeholder(tf.float32, [None, self.image_vector_size])

        with tf.name_scope(self.name):
            # Node and bias
            with tf.name_scope('weights'):
                self.W = weight_variable([self.image_vector_size, 6])
                variable_summaries(self.W)

            with tf.name_scope('biases'):
                self.b = bias_variable([6])
                variable_summaries(self.b)

        # Result
        with tf.name_scope('result'):
            self.y = tf.nn.softmax(tf.matmul(self.x, self.W) + self.b)
            tf.summary.histogram('pre_activation', self.y)

    def train(self, data_set, data_result, train_count, test_part_size, train_cycle, train_batch):
        """
            Entraine le réseau de neuronne
        """
        print("Training on {} data".format(train_count))

        # Expected result y_ and cross_entropy error reduction model
        self.y_ = tf.placeholder(tf.float32, [None, 6])
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y_, logits=self.y))

        # Training
        train_step = tf.train.GradientDescentOptimizer(0.05).minimize(cross_entropy)

        sess = tf.InteractiveSession()
        tf.global_variables_initializer().run()

        data_x, test_x, data_y, test_y = get_test_set(test_part_size, data_set, data_result)
        print("Training set of {} and test set of {} data".format(len(data_x), len(test_x)))

        for i in range(train_cycle):
            batch_x, batch_y = get_training_batch(train_batch, data_x, data_y)
            sess.run(train_step, feed_dict={self.x: batch_x, self.y_: batch_y})

        merged = tf.summary.merge_all()
        self.log(sess.graph)

        correct_predictions = tf.equal(tf.arg_max(self.y, 1), tf.arg_max(self.y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

        print(sess.run(accuracy, feed_dict={self.x: test_x, self.y_: test_y}))

    def log(self, graph):
        file_writer = tf.summary.FileWriter('{}.log'.format(self.name), graph=graph)

    def predict(self):
        """
            Retourne une prédiction
            L'AI doit avoir été entrainée pour produire
            des résultats corrects
        """
        print("Predict")

    def save(self, path):
        """
            Sauvegarde le graphe et l'état du réseau
        """
        print("saving")

    @staticmethod
    def from_save(cls, path):
        """
        Instancie une AI à partir de sa sauvegarde
        :param cls: BernardAi
        :param path: Chemin vers le fichier de sauvegarde
        :return: Instance de BernardAI
        """
        print("Loading")
