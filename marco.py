import pandas as pd 
#from scipy.io import arff 
import math
import numpy as np
import itertools
import Arguments

def unWrapper (filename):
	#opens  arff files and transform to dataFrame
	# rawData = arff.loadarff(filename)
	# dataset = pd.DataFrame(rawData[0])

	dataset = pd.read_csv(filename)

	#discover the target name (is the last atribute in the training set)
	attribute_names = dataset.axes[1]
	target_name= attribute_names[len(attribute_names)-1]

	#get the targets and patterns
	if "forestfires" in filename:
		dbTargets = dataset[ target_name ]
		dbTargets = [ math.log(x) for x in dbTargets]
		dbTargets = pd.Series(dbTargets)

		dbPatterns = dataset.drop( target_name ,axis=1)
		dbPatterns = dbPatterns.drop("month" ,axis=1)

	elif "servo" in filename:
		dbTargets = dataset[ target_name ]

		dbPatterns = dataset.drop( target_name ,axis=1)
		dbPatterns = dbPatterns.drop(attribute_names[0] ,axis=1)
		dbPatterns = dbPatterns.drop(attribute_names[1] ,axis=1)
		#print(dbPatterns)
	else:
		dbTargets = dataset[ target_name ]
		#print("dbTargets",type(dbTargets))
		dbPatterns = dataset.drop( target_name ,axis=1)
		#print("dbPatterns",type(dbPatterns))
	return (dbPatterns,dbTargets)

def getPossibleValues(patterns):
	possibleDataBaseValues = []
	for X in patterns.axes[1]:
		valsInX = patterns[X].unique().tolist()
		possibleDataBaseValues.append(valsInX)
	possibleDataBaseValuesArray = np.array( possibleDataBaseValues )
	#print(possibleDataBaseValuesArray)
	return possibleDataBaseValuesArray

def makeCOBmat(possibleDataBaseValues):
	possibleList = possibleDataBaseValues.tolist()
	cobMat = itertools.product(*possibleList)
	return cobMat

def makeFuzzyProbabilisticFunction(possibleDataBaseValues, patternSet, targetSet):
	fpfX = []
	possibleDataBaseValuesDataFrame = pd.DataFrame(possibleDataBaseValues).T 
	#print("df",possibleDataBaseValuesDataFrame)
	possibleDataBaseValuesDataFrame.columns = patternSet.axes[1]
	#print("ndf",possibleDataBaseValuesDataFrame)

	
	for label in possibleDataBaseValuesDataFrame.axes[1]:
		labelFPF = []
		for val in possibleDataBaseValuesDataFrame[label].values[0]:
			args = Arguments.Arguments()
			args.fitX(label,val,patternSet,targetSet)
			labelFPF.append( args )
		fpfX.append(labelFPF)
	#print("makeFPF",fpf)

	fpfY = Arguments.Arguments()
	fpfY.fitY(targetSet)

	return (fpfX,fpfY)


def path():
	filenames  = ["test.csv"]
	for filename in filenames:
		db_set,db_target = unWrapper(filename)
		possibleDataBaseValues = getPossibleValues(db_set)
		#print( possibleDataBaseValues )
		cobMat = makeCOBmat(possibleDataBaseValues)
		# for i in cobMat:
		# 	print (i)
		fpfX,fpfY = makeFuzzyProbabilisticFunction(possibleDataBaseValues, db_set, db_target )
		print (fpfX, len(fpfX), len(possibleDataBaseValues))
		print (fpfY)


path()