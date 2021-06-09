import knn
import util
import numpy
import random
import typeHinting

Population = tuple[list, float]
Individual = list[float]


def fitnessValue(individual: typeHinting.Individual, trainingSet, testSet) -> float:
    random.seed(2)

    __predictions = []

    for x in range(len(testSet)):
        __neighbors = knn.getNeighbors(
            k=3,
            individual=individual,
            trainingSet=trainingSet,
            testInstance=testSet[x],
        )
        __predictions.append(
            knn.calculate(__neighbors)
        )

    __accuracy = knn.getAccuracy(testSet, __predictions)

    return float(__accuracy)


def tournament(chrom2: list):
    __best = []

    for x in range(2):
        a = numpy.random.randint(0, 7)
        __best.append(chrom2[a])

    __bestOne = __best[0]

    if(__best[0][1] < __best[1][1]):
        __bestOne = __best[1]

    return __bestOne[0]


def mutate(chrom2, probability, next2, n):
    rng = n - len(next2)
    for i in range(rng):
        a = tournament(chrom2)
        rd = []
        for x in range(len(a)):
            if numpy.random.uniform(0.0, 1.0) <= probability:
                rd.append(1)
            else:
                rd.append(0)
        if a == rd:
            a = tournament(chrom2)
        result = util.toXor(a, rd)
        next2.append(result)
    return next2


def crossover(population, probability: float, next2):
    __range = int(probability * len(population))

    for i in range(__range):
        __x1 = tournament(population)
        __x2 = tournament(population)

        if __x1 == __x2:
            next2.append(__x1)
        else:
            next2.append(util.toXor(__x1, __x2))

    return next2
