import ga
import os
import time
import util
from flask import Flask
from flask import request

app = Flask(__name__)

@app.route("/api/recalculate")
def test():
    __splitProbability = float(request.args.get(key='probability', default='0.8', type=float))
    __path = request.args.get('path', '/data/wsd.csv')

    return calculate(
        path=__path,
        splitProbability=__splitProbability
    )

def main():
    calculate()

def calculate(path='/data/wsd.csv', splitProbability=0.7):
    __startTime = time.time()

    __trainingSet, __testSet = util.loadDataset(
        filename=os.path.dirname(__file__)+path,
        splitProbability=splitProbability
    )

    # first individu
    __knnIndividual = [1.0] * (len(__trainingSet[0]) - 1)

    __kValue = 3

    __startTimeWithoutGa = time.time()
    __knnAccuracy = ga.fitnessValue(
        individual=__knnIndividual,
        trainingSet=__trainingSet,
        testSet=__testSet,
        k=__kValue
    )
    __totalTimeWithoutGa = (time.time() - __startTimeWithoutGa)

    # generation number change to population count, this mean
    __startTimeWithGa = time.time()
    __populationSize = 15
    __iterationLimit = 20
    __stall = __iterationLimit - 5
    __counter = 0
    __newFitness = 0.0
    __oldFitness = 0.0

    __messages = []

    __populationWithFitness = ga.generatePopulation(
        k=__kValue,
        testSet=__testSet,
        trainingSet=__trainingSet,
        populationSize=__populationSize,
    )

    for x in range(__iterationLimit):
        # save previous fitness
        __oldFitness = __newFitness

        newChrom, __newFitness, bestFeature = ga.evolve(
            population=__populationWithFitness,
            index=x,
            genNum=__populationSize
        )

        # __messages.append(__message)

        __populationWithFitness = []

        for y in range(__populationSize):
            __individual = newChrom[y]

            __populationWithFitness.append([
                __individual,
                ga.fitnessValue(
                    individual=__individual,
                    trainingSet=__trainingSet,
                    testSet=__testSet,
                    k=__kValue,
                )
            ])

        if (__newFitness - __oldFitness) <= 0.0000001:
            __counter += 1
        else:
            __counter = 0

        if __counter == __stall:
            break

    # Ensure virus_detected (label) is not modified to 0
    # bestFeature.pop(39)

    __totalTimeWithGa = (time.time() - __startTimeWithGa)

    print("")
    print("Accuracy K-NN without GA is {:.3f}% with total time {:.3f}s".format(
        float(__knnAccuracy), __totalTimeWithoutGa
    ))
    print("Accuracy K-NN with GA is \033[92m{:.3f}%\033[0m with total time {:.3f}s".format(
        float(__newFitness), __totalTimeWithGa
    ))
    print("")
    print("Best feature is", bestFeature)
    print("")
    print("\033[92mExecution completed after {:.3f}s".format(
        (time.time() - __startTime)
    ))

    return {
        "message": "Successfully recalculate data",
        "execution_time": (time.time() - __startTime),
        "population_size": __populationSize,
        "maximum_regeneration": __iterationLimit,
        "generation_progress": __messages,
        "best_feature": bestFeature,
        "accuracy": {
            "without_ga": float(__knnAccuracy),
            "with_ga": float(__newFitness)
        },
    }


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('')
        exit(0)
