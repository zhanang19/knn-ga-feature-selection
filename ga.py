import knn
import numpy
import sys
import os


def generatePopulation(n):
    __originalChromosome = []
    for x in range(n):
        __temp = []
        __temp2 = []
        for x in range(10):
            __temp.append(numpy.random.randint(1))
        __temp2.append(__temp)
        fitness = knn.fitnessValue(__temp)
        __temp2.append(fitness)
        __originalChromosome.append(__temp2)
    return __originalChromosome


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
        result = toXor(a, rd)
        next2.append(result)
    return next2


def toXor(x1: list, x2: list) -> list:
    __xorResult = []
    for __i in range(len(x1)):
        if x1[__i] == x2[__i]:
            __xorResult.append(0)
        else:
            __xorResult.append(1)
    return __xorResult


def getCrossOver(chrom2: list, probability: float, next2):
    rng = int(probability * len(chrom2))
    for x in range(rng):
        a = tournament(chrom2)
        b = tournament(chrom2)
        if a == b:
            result = a
        else:
            result = toXor(a, b)
        next2.append(result)
    return next2


def getGeneration(chrom2, index, genNum):
    chrom2, nextGen, bestFitness, bestFt2 = eliteChild(chrom2, genNum)
    nextGen = getCrossOver(chrom2, 0.8, nextGen)
    nextGen = getMutation(chrom2, 0.3, nextGen, genNum)
    print('Generation {} {:.3f}%'.format(index + 1, float(bestFitness)))
    return nextGen, bestFitness, bestFt2

def main():

    genNum = 15
    knn_chrom = [0, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    knn_acc = knn.fitnessValue(knn_chrom)
    limit = 120
    stall = 50
    chrom = generatePopulation(n=genNum)
    counter = 0
    newFit = 0.0
    oldFit = 0.0
    for x in range(limit):
        oldFit = newFit
        newChrom, newFit, bestFt = getGeneration(chrom, x, genNum=genNum)
        newFit = float(newFit)
        temp2 = []
        for y in range(genNum):
            temp = []
            temp.append(newChrom[y])
            f = knn.fitnessValue(newChrom[y])
            temp.append(f)
            temp2.append(temp)
        chrom = temp2
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
