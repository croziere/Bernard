"""
Fonctions d'aide pour l'affichage sur tensorflow
"""
import tensorflow as tf


def variable_summaries(var):
    """
    Attache les données à un Tensor
    Permet la visualisation sur tensorboard de :
    La moyenne
    Le max
    Le min
    L'histogramme
    """
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)
