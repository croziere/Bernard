import io
import random

import numpy
import tensorflow as tf
import sys
from PIL import Image, ImageOps
from ai import BernardAi

from network import *
from train import split_data_set


def compile_config(argv, config):
    return config.update(argv)


def create_training_set(path):
    data_set = []
    i = 0
    with open(path, 'rb') as file:
        while True:
            i = i + 1
            entry = {'data': None, 'label': None}
            img = file.read(687126)
            if not img: break
            result = file.read(1)
            res_array = [0, 0, 0, 0, 0, 0]
            res_array[int.from_bytes(result, 'big')] = 1
            entry['label'] = res_array
            image = Image.frombytes('RGB', (638, 359), img)
            image = image.convert('L')
            image = image.rotate(180)
            image.thumbnail([64, 36], Image.ANTIALIAS)
            image = ImageOps.mirror(image)
            image.save('images/img{}-{}.bmp'.format(i, int.from_bytes(result, 'big')))
            entry['data'] = numpy.array(image).ravel()
            data_set.append(entry)
    return data_set


def main():
    data_set = create_training_set('DataSet')

    print("Dataset of {} images".format(len(data_set)))

    train_set, test_set = split_data_set(0.5 * len(data_set), data_set)

    ai = BernardAi("Bernard-II")
    ai.init(64, 36, 6)

    ai.train(train_set, test_set, 1, 5)
    print(ai.predict(test_set))


if __name__ == "__main__":
    main()
