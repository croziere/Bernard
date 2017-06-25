"""
Fonctions d'aide pour l'entrainement du réseau
"""
import random

import numpy
import tensorflow as tf


def get_batch(data_set):
    """
    Retourne un set de données sous la forme d'une vecteur x (input) et sortie (y)
    :param data_set: Le jeu de données
    :return: batch[0] données x batch[1] données y
    """
    batch = [[], []]
    for i in range(len(data_set)):
        batch[0].append(data_set[i]['data'])
        batch[1].append(data_set[i]['label'])

    return batch

def get_random_batch(n, data_set):
    """
    Créé un set de données aléatoire de taille n parmis data_set
    :param n: Taille du set
    :param data_set: Le set de données
    :return: batch[0] Les données x  batch[1] les données y
    """
    indexes = random.sample(range(len(data_set)), n)

    batch = [[], []]
    for i in indexes:
        batch[0].append(data_set[i]['data'])
        batch[1].append(data_set[i]['label'])

    return batch


def split_data_set(n, data_set):
    """
    Coupe le jeu de données au n-ième index
    :param n: L'index ou effectuer la séparation
    :param data_set: Le jeu de données
    """
    n = int(n)
    test_set = data_set[:n]
    train_set = data_set[n:]

    return train_set, test_set
