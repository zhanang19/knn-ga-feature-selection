import knn
import ga
import numpy
import sys
import os
import util
from sklearn.preprocessing import MinMaxScaler

# Type alias
Dataset = tuple[list, list]
Population = tuple[list, float]


def generatePopulation(n: int, trainingSet: list, testSet: list) -> Population:
    __population = []
    for a in range(n):
        __individual = []

        for x in range(len(trainingSet[0]) - 1):
            __individual.append(numpy.random.randint(1))

        __population.append([
            __individual,
            ga.fitnessValue(
                population=__individual,
                trainingSet=trainingSet,
                testSet=testSet,
            )
        ])

    return __population


def eliteChild(chrom2: list, n: int):
    a = sorted(chrom2, key=lambda l: l[1], reverse=True)
    bestVal = a[0][1]
    bestFt2 = a[0][0]
    next2 = []
    chrom3 = []

    for x in range(2):
        next2.append(a[x][0])

    for x in range(2, n):
        chrom3.append(a[x])

    return chrom3, next2, bestVal, bestFt2


def tournament(chrom2: list):
    __best = []

    for x in range(2):
        a = numpy.random.randint(0, 7)
        __best.append(chrom2[a])

    __bestOne = __best[0]

    if(__best[0][1] < __best[1][1]):
        __bestOne = __best[1]

    return __bestOne[0]


def getMutation(chrom2, probability, next2, n):
    rng = n - len(next2)
    for i in range(rng):
        a = tournament(chrom2)
        rd = []
        for x in range(10):
            if numpy.random.uniform(0.0, 1.0) <= probability:
                rd.append(1)
            else:
                rd.append(0)
        if a == rd:
            a = tournament(chrom2)
        result = util.toXor(a, rd)
        next2.append(result)
    return next2


def getCrossOver(population: Population, probability: float, next2):
    __range = int(probability * len(population))

    for i in range(__range):
        __x1 = tournament(population)
        __x2 = tournament(population)

        if __x1 == __x2:
            next2.append(__x1)
        else:
            next2.append(util.toXor(__x1, __x2))

    return next2


def getGeneration(population: Population, index, genNum):
    population, nextGen, bestFitness, bestFt2 = eliteChild(population, genNum)
    nextGen = getCrossOver(population, 0.8, nextGen)
    nextGen = getMutation(population, 0.3, nextGen, genNum)
    print('Generation {} {:.3f}%'.format(index + 1, float(bestFitness)))
    return nextGen, bestFitness, bestFt2

def main():
    genNum = 15
    knn_chrom = [0, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    __trainingSet, __testSet = util.loadDataset(
        filename='data/glass.csv',
        splitProbability=0.8
    )
    knn_acc = ga.fitnessValue(
        population=knn_chrom,
        trainingSet=__trainingSet,
        testSet=__testSet,
    )
    limit = 120
    stall = 50
    population = generatePopulation(
        trainingSet=__trainingSet,
        testSet=__testSet,
        n=genNum
    )

    counter = 0
    newFit = 0.0
    oldFit = 0.0

    for x in range(limit):
        oldFit = newFit
        newChrom, newFit, bestFt = getGeneration(population, x, genNum=genNum)
        newFit = float(newFit)
        temp2 = []
        for y in range(genNum):
            temp = []
            temp.append(newChrom[y])
            fitness = ga.fitnessValue(
                newChrom[y],
                trainingSet=__trainingSet,
                testSet=__testSet,
            )
            temp.append(fitness)
            temp2.append(temp)

        population = temp2

        diff = newFit - oldFit

        if diff <= 0.0000001:
            counter += 1
        else:
            counter = 0

        if counter == stall:
            break

    print("Accuracy using K-NN without GA: {:.3f}%".format(float(knn_acc)))
    print("Accuracy using K-NN with GA: {:.3f}%".format(float(newFit)))
    print("Used features:", bestFt)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
