import csv
import random
import typeHinting
from sklearn.preprocessing import MinMaxScaler


def loadDataset(filename: str, splitProbability: float) -> typeHinting.Dataset:
    with open(file=filename, mode='rt') as csvfile:
        lines = csv.reader(csvfile)
        dataset = list(lines)

        dataset2 = dataset
        __scaler = MinMaxScaler()
        __scaler.fit(dataset)

        MinMaxScaler(copy=True, feature_range=(0, 1))
        dataset = __scaler.transform(dataset)

        __trainingSet = []
        __testSet = []

        __labelIndex = len(dataset[0]) - 1
        __individualLength = len(dataset[0]) - 2

        for a in range(len(dataset)):
            dataset[a][__labelIndex] = dataset2[a][__labelIndex]

        for x in range(0, len(dataset)):
            for y in range(__individualLength):
                dataset[x][y] = float(dataset[x][y])
            if random.random() < splitProbability:
                __trainingSet.append(dataset[x])
            else:
                __testSet.append(dataset[x])

    return __trainingSet, __testSet

def toXor(male: typeHinting.Individual, female: typeHinting.Individual) -> typeHinting.Individual:
    __child = []
    for __i in range(len(male)):
        if male[__i] == female[__i]:
            __child.append(0)
        else:
            __child.append(1)
    return __child
