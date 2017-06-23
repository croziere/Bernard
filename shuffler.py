import random

import numpy


def main():
    i = 0
    data_set = []
    with open('RawSet', 'rb') as file:
        while True:
            i = i + 1
            entry = {'data': None, 'label': None}
            img = file.read(687126)
            if not img: break
            result = file.read(1)
            entry['label'] = result
            entry['data'] = numpy.array(img).ravel()
            data_set.append(entry)

    random.shuffle(data_set)
    with open('DataSet', 'wb') as file:
        for i in range(len(data_set)):
            file.write(data_set[i]['data'])
            file.write(data_set[i]['label'])


if __name__ == "__main__":
    main()
