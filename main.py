import io

import numpy
import tensorflow as tf
import sys
from PIL import Image, ImageOps
from ai import BernardAi

from network import *


def compile_config(argv, config):
    return config.update(argv)


def create_training_set(path, test_set, test_result):
    i = 0

    with open(path, 'rb') as file:
        while True:
            img = file.read(687126)
            if not img: break
            i = i + 1
            print(i)
            result = file.read(1)
            res_array = [0, 0, 0, 0, 0, 0]
            print(result)
            res_array[int.from_bytes(result, 'big') - 1] = 1
            test_result.append(res_array)
            image = ImageOps.mirror((Image.frombytes('RGB', (638, 359), img)).convert('L').rotate(180))
            test_set.append(numpy.array(image).ravel())


def main():
    test_set = []
    test_result = []

    create_training_set('DataSet', test_set, test_result)

    ai = BernardAi("Bernardo")
    ai.init(638, 359)
    ai.train(test_set, test_result, len(test_result))
    # ai.save()

    """config = {
        "mode": "base"
    }

    #compile_config(sys.argv, config)

    print(config)
    #Image input
    x = tf.placeholder(tf.float32, [None, 100])

    #Node and bias
    W = weight_variable([100, 6])
    b = bias_variable([6])

    #Result
    y = tf.nn.softmax(tf.matmul(x, W) + b)

    #Expected result y_ and cross_entropy error reduction model
    y_ = tf.placeholder(tf.float32, [None, 6])
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

    #Training
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

    #Session
    sess = tf.InteractiveSession()

    tf.global_variables_initializer().run()"""


if __name__ == "__main__":
    main()
