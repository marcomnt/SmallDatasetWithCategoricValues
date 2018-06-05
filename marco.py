import pandas as pd 
#from scipy.io import arff 
import math
import numpy as np
import itertools

def unWrapper (filename):
	#opens  arff files and transform to dataFrame
	# rawData = arff.loadarff(filename)
	# dataset = pd.DataFrame(rawData[0])

	dataset = pd.read_csv(filename)

	#discover the target name (is the last atribute in the training set)
	attribute_names = dataset.axes[1]
	target_name= attribute_names[len(attribute_names)-1]

	#get the targets and patterns
	dbTargets = dataset[ target_name ]
	dbPatterns = dataset.drop( target_name ,axis=1)

	return (dbPatterns,dbTargets)

def getPossibleValues(database):
	possibleDataBaseValues = []
	for X in database.axes[1]:
		valsInX = database[X].unique().tolist()
		possibleDataBaseValues.append(valsInX)
	possibleDataBaseValuesArray = np.array( possibleDataBaseValues )
	return possibleDataBaseValuesArray

def makeCOBmat(possibleDataBaseValues):
	possibleList = possibleDataBaseValues.tolist()
	cobMat = itertools.product(*possibleList)
	return cobMat

def makeFuzzyProbabilisticFunction(possibleDataBaseValues, database):
	pass
	return 0


def path():
	filenames  = ["chess.csv"]
	for filename in filenames:
		db_set,set_target = unWrapper(filename)
		possibleDataBaseValues = getPossibleValues(db_set)
		cobMat = makeCOBmat(possibleDataBaseValues)
		fuzzyProbabilisticFunction = makeFuzzyProbabilisticFunction(possibleDataBaseValues, db_set )
		


path()