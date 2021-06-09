import knn
import util
import numpy
import random
import typeHinting


def generatePopulation(n: int, trainingSet: list, testSet: list) -> typeHinting.PopulationWithFitness:
    __population = []
    for a in range(n):
        __individual = []

        for x in range(len(trainingSet[0]) - 1):
            __individual.append(numpy.random.randint(1))

        __population.append([
            __individual,
            fitnessValue(
                individual=__individual,
                trainingSet=trainingSet,
                testSet=testSet,
            )
        ])

    return __population


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


def crossover(population: list[typeHinting.IndividualWithFitness], probability: float, nextGeneration: typeHinting.Individual) -> typeHinting.Individual:
    __range = int(probability * len(population))

    for i in range(__range):
        __x1 = tournament(population)
        __x2 = tournament(population)

        if __x1 == __x2:
            nextGeneration.append(__x1)
        else:
            nextGeneration.append(util.toXor(__x1, __x2))

    return nextGeneration


def eliteChild(chrom2: typeHinting.PopulationWithFitness, n: int) -> typeHinting.EliteChild:
    a = sorted(chrom2, key=lambda l: l[1], reverse=True)
    bestVal = float(a[0][1])
    bestFeature: typeHinting.Feature = a[0][0]
    nextGeneration = []
    chrom3 = []

    for x in range(2):
        nextGeneration.append(a[x][0])

    for x in range(2, n):
        chrom3.append(a[x])

    return chrom3, nextGeneration, bestVal, bestFeature


def evolve(population, index, genNum) -> tuple[typeHinting.Population, list, typeHinting.Feature]:
    population, nextGeneration, bestFitness, bestFeature = eliteChild(population, genNum)

    nextGeneration = crossover(
        population,
        probability=0.8,
        nextGeneration=nextGeneration
    )

    nextGeneration = mutate(population, 0.3, nextGeneration, genNum)

    print('Generation {} {:.3f}%'.format(index + 1, float(bestFitness)))

    return nextGeneration, bestFitness, bestFeature
