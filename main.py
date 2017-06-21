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

    with open(path, 'rb') as file:
        while True:
            img = file.read(687126)
            if not img: break
            result = file.read(1)
            res_array = [0, 0, 0, 0, 0, 0]
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
    ai.train(test_set, test_result, len(test_result), 10, 10, 4)
    # ai.save()


if __name__ == "__main__":
    main()
