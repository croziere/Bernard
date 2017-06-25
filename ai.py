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
        """
            :param name: Le nom del'AI (utilisé pour créer les fichier de sauvegarde)
        """
        self.name = name
        self.x = None
        """ Vecteur d'entré """

        self.W = None
        """ Poids des neurones """

        self.b = None
        """ Biais """

        self.y = None
        """ Vecteur de sortie du réseau """

        self.y_ = None
        """ Réponses attendues pour l'entrainement """

        self.sumpath = '{}-summary'.format(name)
        """ Dossier de sauvegarde """

    def init(self, height, width, output):
        """
        Initialise le réseau de neurones (créé le graphe)
        :param height: Hauteur des images
        :param width: Largeur des images
        :param output: Taille du vecteur de sortie
        """
        print("Initializing {} AI".format(self.name))

        self.image_vector_size = height * width

        self.output = output

        # Vecteur d'entré
        self.x = tf.placeholder(tf.float32, [None, self.image_vector_size])

        x_image = tf.reshape(self.x, [-1, height, width, 1])

        # 1 convolution
        h_pool1 = nn_conv_layer(x_image, [5, 5, 1, 32], 32, "Layer_1")

        # 2 convilution
        h_pool2 = nn_conv_layer(h_pool1, [5, 5, 32, 64], 64, "Layer_2")

        # FC1
        h_pool2_flat = tf.reshape(h_pool2, [-1, 16 * 9 * 64])
        h_fc1 = nn_layer(h_pool2_flat, 16 * 9 * 64, 1024, "FC1")

        self.keep_prob = tf.placeholder(tf.float32)

        # Drop de FC1
        h_fc1_drop = tf.nn.dropout(h_fc1, self.keep_prob)

        # Sortie du réseau
        self.y = nn_layer(h_fc1_drop, 1024, self.output, "FC2")

        # Résultat attendu
        self.y_ = tf.placeholder(tf.float32, [None, self.output])

        with tf.name_scope('cross_entropy'):
            diff = tf.nn.softmax_cross_entropy_with_logits(labels=self.y_, logits=self.y)
            with tf.name_scope('total'):
                cross_entropy = tf.reduce_mean(diff)
        tf.summary.scalar('cross_entropy', cross_entropy)

        # Entrainement
        with tf.name_scope('train'):
            self.train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

        # Calcul de la précision
        with tf.name_scope('accuracy'):
            with tf.name_scope('correct_prediction'):
                correct_predictions = tf.equal(tf.arg_max(self.y, 1), tf.arg_max(self.y_, 1))
            with tf.name_scope('accuracy'):
                self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
        tf.summary.scalar('accuracy', self.accuracy)

        self.sess = tf.InteractiveSession()

        self.merged = tf.summary.merge_all()

        self.train_writer = tf.summary.FileWriter(self.sumpath + '/train', self.sess.graph)
        self.test_writer = tf.summary.FileWriter(self.sumpath + '/test')
        tf.global_variables_initializer().run()

    def train(self, train_set, test_set, train_cycle, train_batch_size):
        """
            Entraine le réseau de neuronne
            :param train_set: Jeu de données d'Entrainement
            :param test_set: Jeu de données de test
            :param train_cycle: Nombre de cycle d'entrainement
            :param train_batch_size: Nombre d'image d'entrainement à chaque cycle
        """
        train_size = len(train_set)
        print("Training {} on {} data {} times with batches of size {}".format(self.name, train_size, train_cycle,
                                                                               train_batch_size))
        for i in range(train_cycle):
            if i % 10 == 0:
                self.test(test_set, i)
            else:
                batch = get_random_batch(train_batch_size, train_set)
                summary, _ = self.sess.run([self.merged, self.train_step],
                                           feed_dict={self.x: batch[0], self.y_: batch[1], self.keep_prob: 0.5})
                self.train_writer.add_summary(summary, i)

        self.save(self.sess)

    def test(self, test_set, i=0):
        """
            Test le réseau et affiche sa précision
            :param test_set: Jeu de test
            :param i: Cycle d'apprentissage
        """
        print("Testing {} with {} test data".format(self.name, len(test_set)))

        batch = get_batch(test_set)
        summary, acc = self.sess.run([self.merged, self.accuracy], feed_dict={self.x: batch[0], self.y_: batch[1], self.keep_prob: 1.0})
        self.test_writer.add_summary(summary, i)
        print('Accuracy at step {}: {}'.format(i, acc))

    def predict(self, image_vector):
        """
            Retourne une prédiction
            L'AI doit avoir été entrainée pour produire
            des résultats corrects
            :param image_vector: Le vecteur d'image sur lequel prédire
        """
        batch = get_batch(image_vector)
        return self.sess.run(self.y, feed_dict={self.x: batch[0], self.keep_prob: 1.0})

    def save(self, sess):
        """
            Sauvegarde le graphe et l'état du réseau
            :param sess: Session à sauvegarder
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
        ai.init(638, 359, 6)
        saver = tf.train.import_meta_graph(path)
        saver.restore(ai.sess, path)
