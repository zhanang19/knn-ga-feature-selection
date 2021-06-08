import csv
import random
import math
import operator

# Type alias
Dataset = tuple[list, list]


def euclideanDistance(firstInstance: list, secondInstance: list, chromosome: list) -> float:
    __distance = 0

    for x in range(1, len(firstInstance) - 1):
        if chromosome[x] == 1:
            __distance += pow((float(firstInstance[x]) -
                              float(secondInstance[x])), 2)

    return math.sqrt(__distance)


def getNeighbors(trainingSet: list, testInstance: list, k: int, chromosome: list) -> list:
    __distances = []

    for __i in range(len(trainingSet)):
        __currentDistance = euclideanDistance(
            firstInstance=testInstance,
            secondInstance=trainingSet[__i],
            chromosome=chromosome
        )
        __distances.append((trainingSet[__i], __currentDistance))

    __distances.sort(key=operator.itemgetter(1))
    __neighbors = []

    for __i in range(k):
        __neighbors.append(__distances[__i][0])

    return __neighbors


def getResponse(neighbors: list) -> float:
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

    return float(__sortedVotes[0][0])


def getAccuracy(testSet: list, predictions: list):
    __correct = 0
    for x in range(len(testSet)):
        if testSet[x][-1] == predictions[x]:
            __correct += 1

    return (__correct / float(len(testSet))) * 100.0
