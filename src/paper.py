import pandas as pd 
#from scipy.io import arff 
from copy import deepcopy
import math
import numpy as np
import itertools
import Arguments
import random

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

def combine(cobMat,virtualValues):
	rawData = itertools.product(cobMat,virtualValues)
	return rawData

def makeVirtualValues( probFuzzyArgs , m):
	random.seed()
	virtualValues = []

	for i in range(m):
		certified = False
		while not certified:
			# print("Lower", probFuzzyArgs.L, "Upper", probFuzzyArgs.U)
			vx = random.uniform( probFuzzyArgs.L , probFuzzyArgs.U)
			# print( "vx", vx)
			value = probFuzzyArgs.calculate(vx)
			# print( "value", value)
			rs = random.uniform(0.0,1.0)
			# print("rs", rs)
			if value>rs:
				certified = True

		# print("\n VEIO UM \n")
		virtualValues.append(vx)

	return virtualValues
			
def probabilityCalculus( data , probFuzzyArgs):
	rawDataProb = []

	for line in data:
		mult = 1
		pattern, target = line[0], line[1]
		sizePattern = len(pattern)
		for index in range(sizePattern):
			fuzzy = searchValue(pattern[index], probFuzzyArgs, index)
			# print( "Index", index)
			# print("Lower Bound", fuzzy.L)
			# print("Center", fuzzy.C)
			# print("Upper Bound", fuzzy.U)
			# print("Target", target)
			# print("Calculate", fuzzy.calculate(target))
			mult = mult * fuzzy.calculate(target)

		if(sizePattern == 0):
			proba = 0
		else:
			proba = math.pow(mult,1/sizePattern )

		rawDataProbLine = list(pattern)
		rawDataProbLine.append(target)
		rawDataProbLine.append(proba)
		rawDataProb.append(rawDataProbLine)


	return rawDataProb

def searchValue(value, probFuzzyArgs, index):
	for arg in probFuzzyArgs[index]:
		if (arg.value==value):
			return arg
	print("ERRO: valor nao encontrado")
	return 0


def alphaCut(patternSet, targetSet, probabilities, alpha):
	
	indexes = []
	for i in range(len(probabilities)):
		if probabilities[i] >= alpha:
			indexes.append(i)

	return (patternSet[indexes,:],targetSet[indexes])

def path():
	# print("START")
	filenames  = ['../datasets/test.csv']
	for filename in filenames:
		db_set,db_target = unWrapper(filename)
		train_set = db_set #usando todo banco de dados para treinamento
		train_target = db_target

		possibleDataBaseValues = getPossibleValues(train_set)
		# print( possibleDataBaseValues )

		cobMat = makeCOBmat(possibleDataBaseValues)
		# for i in deepcopy(cobMat):
		# 	print (i)

		fpfX,fpfY = makeFuzzyProbabilisticFunction(possibleDataBaseValues, train_set, train_target )
		#print (fpfX, len(fpfX), len(possibleDataBaseValues))
		#print (fpfY)

		# for Xi in fpfX:
		# 	print ("Xi")
		# 	print("Valor |          L          |         uSet         |          U          |     MFvalue    ")
		# 	for args in Xi:
		# 		print(args.value, "    |", args.L, "|", args.C, "|", args.U, "|", args.calculate(0.8))

		# PAPER: m = 100%, 200%, 300%, 400% e 500% relativo ao tamanho do conjunto de treinamento
		sizesOfM=[1,2,3,4,5]
		for sizeM in sizesOfM:
			virtualValues = makeVirtualValues( fpfY, m = sizeM*train_set.shape[0])
			# print(virtualValues)

			rawData = combine(deepcopy(cobMat),virtualValues)

			# for i in deepcopy(rawData):
			# 	print (i)

			rawDataProb = probabilityCalculus(deepcopy(rawData), fpfX)
			
			df = pd.DataFrame(rawDataProb)
			# print ("RAW\n",df.values)
			# print("------------")
			sizeDf = df.shape[1]
			patternSet = df.iloc[:,0:sizeDf-2]
			targetSet = df.iloc[:,sizeDf-2:sizeDf-1]
			probabilities = df.iloc[:,sizeDf-1:sizeDf]

			newPattern,newTargets = alphaCut(patternSet.values , targetSet.values , probabilities.values , alpha = 0.7)
			print(newPattern,newTargets)

path()