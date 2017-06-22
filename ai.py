"""
    Ai Module
    Contains all the logic to create a playing AI
"""

import tensorflow as tf
from network import *
from summary import variable_summaries
from train import get_random_batch, get_batch


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

    def init(self, height, width, output):
        """
        Initialise le réseau de neurones
        """
        print("Initializing {} AI".format(self.name))

        self.image_vector_size = height * width
        self.output = output

        # Image input
        self.x = tf.placeholder(tf.float32, [None, self.image_vector_size])

        with tf.name_scope(self.name):
            # Node and bias
            with tf.name_scope('weights'):
                self.W = weight_variable([self.image_vector_size, self.output])
                variable_summaries(self.W)

            with tf.name_scope('biases'):
                self.b = bias_variable([self.output])
                variable_summaries(self.b)

        # Result
        with tf.name_scope('result'):
            self.y = tf.nn.softmax(tf.matmul(self.x, self.W) + self.b)
            tf.summary.histogram('pre_activation', self.y)

        self.sess = tf.InteractiveSession()

    def train(self, train_set, train_cycle, train_batch_size, gradient):
        """
            Entraine le réseau de neuronne
        """
        train_size = len(train_set)
        print("Training {} on {} data {} times with batches of size {} with {} gradient".format(self.name, train_size, train_cycle, train_batch_size, gradient))

        # Expected result y_ and cross_entropy error reduction model
        self.y_ = tf.placeholder(tf.float32, [None, self.output])
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y_, logits=self.y))

        # Training
        train_step = tf.train.GradientDescentOptimizer(gradient).minimize(cross_entropy)

        tf.global_variables_initializer().run()

        for _ in range(train_cycle):
            batch = get_random_batch(train_batch_size, train_set)
            self.sess.run(train_step, feed_dict={self.x: batch[0], self.y_: batch[1]})


        merged = tf.summary.merge_all()
        self.log(self.sess.graph)
        self.save(self.sess)

    def test(self, test_set):
        print("Testing {} with {} test data".format(self.name, len(test_set)))
        correct_predictions = tf.equal(tf.arg_max(self.y, 1), tf.arg_max(self.y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

        batch = get_batch(test_set)
        print(self.sess.run(accuracy, feed_dict={self.x: batch[0], self.y_: batch[1]}))

    def log(self, graph):
        file_writer = tf.summary.FileWriter('{}.log'.format(self.name), graph=graph)

    def predict(self, image_vector):
        """
            Retourne une prédiction
            L'AI doit avoir été entrainée pour produire
            des résultats corrects
        """
        return self.sess.run(self.y, feed_dict={self.x: image_vector})

    def save(self, sess):
        """
            Sauvegarde le graphe et l'état du réseau
        """
        saver = tf.train.Saver()
        savePath = 'save/{}'.format(self.name)
        saver.save(sess, savePath)
        saver.export_meta_graph(savePath)
        print("{} sauvegardé dans {}".format(self.name, savePath))


    @staticmethod
    def from_save(path, name):
        """
        Instancie une AI à partir de sa sauvegarde
        :param cls: BernardAi
        :param path: Chemin vers le fichier de sauvegarde
        :return: Instance de BernardAI
        """
        print("Loading")
        ai = BernardAi(name)
        ai.init(638, 359)
        saver = tf.train.import_meta_graph(path)
        saver.restore(ai.sess, path)

