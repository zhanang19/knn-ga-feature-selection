Feature = list[int]

Individual = list[float]
IndividualWithFitness = list[Individual, float]

Population = list[Individual]
PopulationWithFitness = list[IndividualWithFitness]

EliteChild = list[list, Population, float, Feature]
Generation = tuple[Population, float]

TrainingSet = list[float]
TestSet = list[float]
Dataset = tuple[TrainingSet, TestSet]
