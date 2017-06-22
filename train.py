import random

import numpy
import tensorflow as tf


def get_batch(data_set):
    """
    Return a full batch from a dataSet
    :param data_set:
    :return: batch[0] With input data and batch[1] with output data
    """
    batch = [[], []]
    for i in range(len(data_set)):
        batch[0].append(data_set[i]['data'])
        batch[1].append(data_set[i]['label'])

    return batch

def get_random_batch(n, data_set):
    """
    Get a random batch of size n from a dataSet
    :param n: Size of the batch
    :param data_set: The data set
    :return: batch[0] With input data and batch[1] with output data
    """
    indexes = random.sample(range(len(data_set)), n)

    batch = [[], []]
    for i in indexes:
        batch[0].append(data_set[i]['data'])
        batch[1].append(data_set[i]['label'])

    return batch


def split_data_set(n, data_set):
    n = int(n)
    test_set = data_set[:n]
    train_set = data_set[n:]

    return train_set, test_set
