import csv
import random
from sklearn.preprocessing import MinMaxScaler

# Type alias
Dataset = tuple[list, list]
Population = tuple[list, float]
Individual = list[float]

def loadDataset(filename: str, splitProbability: float) -> Dataset:
    with open(filename, 'rt') as csvfile:
        lines = csv.reader(csvfile)
        dataset = list(lines)
        dataset2 = dataset
        __scaler = MinMaxScaler()
        __scaler.fit(dataset)

        MinMaxScaler(copy=True, feature_range=(0, 1))
        dataset = __scaler.transform(dataset)

        __trainingSet = []
        __testSet = []

        for a in range(len(dataset)):
            dataset[a][10] = dataset2[a][10]

        for x in range(0, len(dataset)):
            for y in range(9):
                dataset[x][y] = float(dataset[x][y])
            if random.random() < splitProbability:
                __trainingSet.append(dataset[x])
            else:
                __testSet.append(dataset[x])

    return __trainingSet, __testSet

def toXor(x1: list, x2: list) -> list:
    __xorResult = []
    for __i in range(len(x1)):
        if x1[__i] == x2[__i]:
            __xorResult.append(0)
        else:
            __xorResult.append(1)
    return __xorResult
