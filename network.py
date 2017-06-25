"""
Contient les fonctions permettant de construire plus facielement le réseau
"""
import tensorflow as tf

from summary import variable_summaries


def weight_variable(shape):
    """
    Créé la variable de poid des neurones
    :param shape: Le vecteur de forme de la variable
    """
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    """
    Créé la variable de biais
    :param shape: Le vecteur de forme de la variable
    """
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def nn_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu):
    """
    Créé une couche de réseau de neuronne
    Assigne les scope pour le rendu sur tensorboard
    :param input_tensor: Tensor en entré de la couche
    :param input_dim: Dimension d'entré
    :param output_dim: Dimension de la sortie
    :param layer_name: Nom de la couche pour l'affichage sur tensorboard
    :param act: Type de fonction d'activation du neuronne (RELu par défaut)
    """
    with tf.name_scope(layer_name):
        with tf.name_scope('weights'):
            weights = weight_variable([input_dim, output_dim])
            variable_summaries(weights)
        with tf.name_scope('biases'):
            biases = bias_variable([output_dim])
            variable_summaries(biases)
        with tf.name_scope('Wx_plus_b'):
            preactivate = tf.matmul(input_tensor, weights) + biases
            tf.summary.histogram('pre_activations', preactivate)
        activations = act(preactivate, name='activation')
        tf.summary.histogram('activations', activations)
        return activations

def nn_conv_layer(input_tensor, input_vector, output_dim, layer_name, act=tf.nn.relu):
    """
    Créé une couche de réseau de convolution
    :param input_tensor: Tensor en entré de la couche
    :param input_dim: Vecteur de dimension d'entré
    :param output_dim: Dimension de la sortie
    :param layer_name: Nom de la couche pour l'affichage sur tensorboard
    :param act: Type de fonction d'activation du neuronne (RELu par défaut)
    """
    with tf.name_scope(layer_name):
        with tf.name_scope('weights'):
            weights = weight_variable(input_vector)
            variable_summaries(weights)
        with tf.name_scope('biases'):
            biases = bias_variable([output_dim])
            variable_summaries(biases)
        with tf.name_scope('Wx_plus_b'):
            preactivate = conv2d(input_tensor, weights) + biases
            tf.summary.histogram('pre_activations', preactivate)
        activations = act(preactivate, name='activation')
        tf.summary.histogram('activations', activations)
        pool = max_pool_2x2(activations)
        return pool

def conv2d(x, W):
    """
        Créé le réseau de convolution
        :param x: Tensor d'entrée
        :param W: Tensor des poids des neurones
    """
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    """
    Créé le réseau de pooling pour le réseau de convolution (fonction max)
    :param x: Tensor d'entrée
    """
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
