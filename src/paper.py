import pandas as pd 
#from scipy.io import arff 
from copy import deepcopy
import math
import numpy as np
import itertools
import Arguments
import random
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor as MLP
from sklearn.model_selection import KFold

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
		# dbPatterns = dbPatterns.drop(attribute_names[0] ,axis=1)
		# dbPatterns = dbPatterns.drop(attribute_names[1] ,axis=1)
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
		if(type(possibleDataBaseValuesDataFrame[label].values[0]) != list):
			for val in possibleDataBaseValuesDataFrame[label].values:
				args = Arguments.Arguments()
				args.fitX(label,val,patternSet,targetSet)
				labelFPF.append( args )
			fpfX.append(labelFPF)
		else:
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

def mean_absolute_percentage_error(y_true, y_pred): 
	vector = np.abs((y_true - y_pred) / y_true)
	return (np.mean(vector) * 100, np.std(vector,ddof=1))

def takeBest(lista):
	best = (np.inf,np.inf)
	for i in lista:
		if(i[0]<best[0]):
			best = i
		if(i[0]==best[0]):
			if(i[1]<best[1]):
				best = i

	return best

def takeWorst(lista):
	worst = (-np.inf,-np.inf)
	for i in lista:
		if(i[0]>worst[0]):
			worst = i
		if(i[0]==worst[0]):
			if(i[1]>worst[1]):
				worst = i

	return worst

def path():
	# print("START")
	filenames  = ["servo2.csv"]
	for filename in filenames:
		db_set,db_target = unWrapper(filename)
		k_fold = KFold(n_splits = 10, shuffle=True)
		print("START")
		# PAPER: m = 100%, 200%, 300%, 400% e 500% relativo ao tamanho do conjunto de treinamento
		sizesOfM=[1,2,3,4,5]
		for sizeM in sizesOfM:
			MAPEsvr=[]
			MAPEmlp =[]
			MAPEallSVR=[]
			MAPEallMLP =[]
			for test_index, train_index in k_fold.split(db_set):

				train_set = db_set.iloc[train_index]
				train_target = db_target.iloc [train_index]

				test_set = db_set.iloc[train_index]
				test_target = db_target.iloc[train_index]

				# print("TRAIN SET", train_set)
				# print("TRAIN TARGET", train_target)
				
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
				targetSet = df.iloc[:,sizeDf-2:sizeDf-1][sizeDf-2]
				# print(targetSet, type(targetSet))
				probabilities = df.iloc[:,sizeDf-1:sizeDf]

				newPattern,newTargets = alphaCut(patternSet.values , targetSet.values, probabilities.values , alpha = 0.7)

				# print(newPattern,newTargets)
				# print(train_set.values,train_target.values)

				full_db_pattern= np.concatenate((newPattern,train_set.values))
				full_db_target = np.concatenate((newTargets,train_target.values))

				"""SVR"""
				svr = SVR()
				svr.fit(full_db_pattern,full_db_target)

				""" MLP """
				mlp = MLP(max_iter=2000)
				mlp.fit(full_db_pattern,full_db_target)

				"""predict target SVR"""
				predictionsSVR = svr.predict(test_set)
				mapejSVR = mean_absolute_percentage_error(test_target.values,predictionsSVR)
				#print (mapejSVR)
				MAPEsvr.append(mapejSVR)
				MAPEallSVR.append(mapejSVR[0])

				"""predict target MLP"""
				predictionsMLP = mlp.predict(test_set)
				mapejMLP = mean_absolute_percentage_error(test_target.values,predictionsMLP)
				#print (mapejMLP)
				MAPEmlp.append(mapejMLP)
				MAPEallMLP.append(mapejMLP[0])

			print("\nFile: "+ filename+"\n" )
			MAPEBest = takeBest(MAPEsvr)
			MAPEWorst = takeWorst(MAPEsvr)
			print("------------------------------SVR--------------------------------------")
			print("M: ",sizeM, " MAPE: ", MAPEBest[0], "Standard Deviation:", MAPEBest[1], "BEST")
			print("M: ",sizeM, " MAPE: ", np.mean(MAPEallSVR)," Standard Deviation:", np.std(MAPEallSVR,ddof=1), "MEAN")
			print("M: ",sizeM, " MAPE: ", MAPEWorst[0], "Standard Deviation:", MAPEWorst[1], "WORSE")
			print("-----------------------------------------------------------------------")

			MAPEBest = takeBest(MAPEmlp)
			MAPEWorst = takeWorst(MAPEmlp)
			print("--------------------------------MLP------------------------------------")
			print("M: ",sizeM, " MAPE: ", MAPEBest[0], "Standard Deviation:", MAPEBest[1], "BEST")
			print("M: ",sizeM, " MAPE: ", np.mean(MAPEallMLP)," Standard Deviation:", np.std(MAPEallMLP,ddof=1), "MEAN")
			print("M: ",sizeM, " MAPE: ", MAPEWorst[0], "Standard Deviation:", MAPEWorst[1], "WORSE")
			print("-----------------------------------------------------------------------")

path()