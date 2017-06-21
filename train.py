import random

import numpy
import tensorflow as tf


def get_training_batch(n, data_set, result_set):
    indexes = random.sample(range(len(result_set)), n)

    batch_x = []
    for i in indexes:
        batch_x.append(data_set[i])

    batch_y = []
    for i in indexes:
        batch_y.append(result_set[i])

    return batch_x, batch_y


def get_test_set(n, data_set, result_set):
    indexes = random.sample(range(len(result_set)), n)

    test_set = []
    for i in indexes:
        test_set.append(data_set[i])

    test_result = []
    for i in indexes:
        test_result.append(result_set[i])

    data_x = []
    data_y = []
    for i in range(len(result_set)):
        if not indexes.__contains__(i):
            data_x.append(data_set[i])
            data_y.append(result_set[i])


    return data_x, test_set, data_y, test_result
