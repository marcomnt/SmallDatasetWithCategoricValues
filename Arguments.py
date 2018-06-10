import pandas as pd
import numpy as np
import math

class Arguments:
	"""docstring for Arguments"""
	# classe para guardar argumentos da funcao Fuzzy
	# self.X  guarda a label
	# self.value guarda o valor da label 
	# self.U eh o upperBound em Y
	# self.C eh o centro em Y
	# self.L eh o lowerBound em Y

	def fitX(self, X, value, patternSet, targetSet):
		self.X = X
		self.value = value
		# print ("value",value)
		collection = targetSet[ patternSet[X]==value ]
		# print ("collection",collection)
		maxValue = max(collection)
		# print ("max", maxValue)
		minValue = min(collection)
		# print ("min", minValue)
		uSet = float(minValue + maxValue)/2
		# print( "uset",uSet)
		sx = np.var(collection)
		# print( "sx",sx)
		nU = len( collection[collection>uSet] )
		# print("nu",nU)
		nL = len( collection[collection<uSet] )
		# print("nl",nL)
		if (nU == 0 and nL == 0):
			nU = 1
			nL = 1
		skewU = nU/float(nU+nL)
		skewL = nL/float(nU+nL)


		self.C = float(uSet)
		self.U = uSet + skewU * math.sqrt(-2*sx/nU*math.log(math.pow(10,-20)))
		self.L = uSet - skewL * math.sqrt(-2*sx/nL*math.log(math.pow(10,-20)))
	
	def fitY(self, Y):
		collection = Y
		maxValue = max(collection)
		# print ("max", maxValue)
		minValue = min(collection)
		# print ("min", minValue)
		uSet = float(minValue + maxValue)/2
		# print( "uset",uSet)
		sx = np.var(collection)
		# print( "sx",sx)
		nU = len( collection[collection>uSet] )
		# print("nu",nU)
		nL = len( collection[collection<uSet] )
		# print("nl",nL)
		if (nU == 0 and nL == 0):
			nU = 1
			nL = 1
		skewU = nU/float(nU+nL)
		skewL = nL/float(nU+nL)


		self.C = float(uSet)
		self.U = uSet + skewU * math.sqrt(-2*sx/nU*math.log(math.pow(10,-20)))
		self.L = uSet - skewL * math.sqrt(-2*sx/nL*math.log(math.pow(10,-20)))

	def calculate(self, x):
		ret = 0
		print("centro", self.C)
		if(self.L <= x and x <= self.C):
			ret = (x-self.L)/(self.C - self.L)
			print("if1", ret)
		elif(self.C < x and x <= self.U):
			ret = (self.U - x)/(self.U - self.C)
			print("if2", ret)
		return ret
