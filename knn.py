import csv
import random
import math
import operator
from sklearn.preprocessing import MinMaxScaler


def loadDataset(filename: str, split: float) -> list:
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
            if random.random() < split:
                __trainingSet.append(dataset[x])
            else:
                __testSet.append(dataset[x])

    return __trainingSet, __testSet


def euclideanDistance(firstInstance: list, secondInstance: list, length: int, chrom2: list):
    __distance = 0

    for x in range(1, length):
        if chrom2[x] == 1:
            __distance += pow((float(firstInstance[x]) - float(secondInstance[x])), 2)

    return math.sqrt(__distance)


def getNeighbors(trainingSet, testInstance, k, chrom2):
    distances = []
    length = len(testInstance)-1
    for x in range(len(trainingSet)):
        dist = euclideanDistance(testInstance, trainingSet[x], length, chrom2)
        distances.append((trainingSet[x], dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors


def getResponse(neighbors):
    __classVotes = {}
    for x in range(len(neighbors)):
        response = neighbors[x][-1]
        if response in __classVotes:
            __classVotes[response] += 1
        else:
            __classVotes[response] = 1

    __sortedVotes = sorted(
        __classVotes.items(),
        key=operator.itemgetter(1),
        reverse=True
    )

    return __sortedVotes[0][0]


def getAccuracy(testSet: list, predictions: list):
    __correct = 0
    for x in range(len(testSet)):
        if testSet[x][-1] == predictions[x]:
            __correct += 1

    return (__correct / float(len(testSet))) * 100.0


def fitnessValue(chrom) -> float:
    random.seed(2)
    split = 0.8

    __trainingSet, __testSet = loadDataset('./data/glass.csv', split)

    predictions = []
    __kValue = 3

    for x in range(len(__testSet)):
        neighbors = getNeighbors(__trainingSet, __testSet[x], __kValue, chrom)
        result = getResponse(neighbors)
        predictions.append(result)
    accuracy = getAccuracy(__testSet, predictions)

    return float(accuracy)
