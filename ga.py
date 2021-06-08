import knn
import random


def fitnessValue(population, trainingSet, testSet) -> float:
    random.seed(2)

    __predictions = []

    for x in range(len(testSet)):
        __neighbors = knn.getNeighbors(
            k=3,
            chromosome=population,
            trainingSet=trainingSet,
            testInstance=testSet[x],
        )
        __predictions.append(
            knn.getResponse(__neighbors)
        )

    __accuracy = knn.getAccuracy(testSet, __predictions)

    return float(__accuracy)
