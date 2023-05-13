import random
from sklearn.preprocessing import MinMaxScaler
from pandas import read_csv
from numpy import NaN


def loadDataset(filename, splitProbability):
    dataframe = read_csv(filename)

    # Data cleaning
    dataframe = dataframe.replace('', NaN)
    dataframe.dropna(inplace=True)

    # Convert dataframe to list type
    dataset = dataframe.values.tolist()

    # Data transformation

    # originalDataset variable will preserve
    # original label during data transformation
    originalDataset = dataset

    __scaler = MinMaxScaler(copy=True, feature_range=(0, 1))
    dataset = __scaler.fit_transform(dataset)

    # As the transformation is completed,
    # we will restore the transformed label
    __indexLabel = len(dataset[0]) - 1
    for a in range(len(dataset)):
        dataset[a][__indexLabel] = originalDataset[a][__indexLabel]

    __trainingSet = []
    __testSet = []
    for x in range(len(dataset)):
        # Ensure all column is float based
        for y in range(len(dataset[0])):
            dataset[x][y] = float(dataset[x][y])

        if random.random() < splitProbability:
            __trainingSet.append(dataset[x])
        else:
            __testSet.append(dataset[x])

    return __trainingSet, __testSet

def toXor(male, female):
    __child = []
    for __i in range(len(male)):
        if male[__i] == female[__i]:
            __child.append(0)
        else:
            __child.append(1)
    return __child
