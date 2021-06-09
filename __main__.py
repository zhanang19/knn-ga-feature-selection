import ga
import util


def main():
    __splitProbability = 0.8

    __trainingSet, __testSet = util.loadDataset(
        filename='data/wsd.csv',
        splitProbability=__splitProbability
    )

    __knnIndividual = [0, 1, 1, 1, 1, 1, 1, 1, 1, 1]

    __knnAccuracy = ga.fitnessValue(
        individual=__knnIndividual,
        trainingSet=__trainingSet,
        testSet=__testSet,
    )

    __generationNumber: int = 15
    __iterationLimit: int = 10
    __stall: int = 40
    __counter: int = 0
    __newFitness: float = 0.0
    __oldFitness: float = 0.0

    __population = ga.generatePopulation(
        trainingSet=__trainingSet,
        testSet=__testSet,
        n=__generationNumber
    )

    for x in range(__iterationLimit):
        __oldFitness = __newFitness
        newChrom, __newFitness, bestFeature = ga.evolve(
            population=__population,
            index=x,
            genNum=__generationNumber
        )

        __population = []

        for y in range(__generationNumber):
            __individual = newChrom[y]

            __fitness = ga.fitnessValue(
                individual=__individual,
                trainingSet=__trainingSet,
                testSet=__testSet,
            )

            __population.append([
                __individual,
                __fitness
            ])

        if (__newFitness - __oldFitness) <= 0.0000001:
            __counter += 1
        else:
            __counter = 0

        if __counter == __stall:
            break

    print("Accuracy K-NN without GA: {:.3f}%".format(float(__knnAccuracy)))
    print("Accuracy K-NN with GA: {:.3f}%".format(float(__newFitness)))
    print("Features used:", bestFeature)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('')
        exit(0)
