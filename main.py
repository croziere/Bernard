#!/bin/python

import io
import random

import numpy
import tensorflow as tf
import sys
from PIL import Image, ImageOps
from ai import BernardAi

from network import *
from train import split_data_set



def create_training_set(path):
    """
    Transforme le fichier de données en dictionnaire de données
    data_set[n]['data'] : Vecteur de l'image
    data_set[n]['label'] : Vecteur de réponse
    :param path: Le chemin du fichier
    """
    data_set = []
    i = 0
    with open(path, 'rb') as file:
        while True:
            i = i + 1
            entry = {'data': None, 'label': None}
            img = file.read(687126) #Les images provenant du jeu font 687126 octets
            if not img: break
            result = file.read(1) #Le résultat est sur 1 octet
            res_array = [0, 0, 0, 0, 0, 0]
            res_array[int.from_bytes(result, 'big')] = 1
            entry['label'] = res_array
            # Transformation de l'image (l'image arrive à l'envers du jeu)
            image = Image.frombytes('RGB', (638, 359), img)
            image = image.convert('L') #Conversion en niveaux de gris
            image = image.rotate(180) #Retournement de l'image
            image.thumbnail([64, 36], Image.ANTIALIAS) #Redimensionnement
            image = ImageOps.mirror(image) #Mirroir
            image.save('images/img{}-{}.bmp'.format(i, int.from_bytes(result, 'big'))) #Sauvegarde le jeu de données afin de le vérifier
            entry['data'] = numpy.array(image).ravel() #On écrit le résultat sur un vecteur à une dimension
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
