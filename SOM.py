'''
This file is an implementation of the SOM algorithm
'''

import time
import numpy as np
from pprint import pprint
from sentiment_som import load_data
from w2v import train_word2vec

#Kernel Parameters
size_x=n_gram=3			#size of 1 kernel and convolution parameter
size_y=300

#SOM Map parameters
nKernel_y=100 		#No of kernels on x and y axis on a SOM Map.

#Training parameters
learning_rate=0.1
sigma=0.3
num_iteration=3

#Word2Vec parameters
embedding_dim=300
min_word_count=1
context=10


class SOM_Map_Layer1:
	def __init__(self,nKernel_y,size_y,size_x,learning_rate=0.1,sigma=0.3,num_iteration=10000):
		self.nKernel_y=nKernel_y
		self.size_y=size_y
		self.size_x=size_x
		self.learning_rate=learning_rate
		self.sigma=sigma

		#Total no. of iterations for the model
		self.num_iteration=num_iteration

		#Load input data matrices
		print "In Init"
		self.x_train, self.y_train, self.x_test, self.y_test, self.vocabulary_inv = load_data()

		#Set random weights for SOM Map
		print "Intializing kernels of the Self Organizing Map"
		self.map=self.setRandomWeights()

	def setRandomWeights(self):
		'''
		Returns a map of Randomly initiated weights
		'''
		kernels=np.array([self.get_kernel(size_x,size_y) for i in range(nKernel_y)])
		return kernels

	def get_kernel(self,size_y,size_x):
		'''
		Returns a kernel of size_y rows and size_x columns
		'''
		kernel=np.random.random_sample((size_y,size_x))
		return kernel

	def getEuclideanDistance(self,a,b):
		'''
		Returns the Euclidean Distance between 2 Matrices
		'''
		return np.linalg.norm(a-b)

	def findBestMatchingUnit(self,inp):
		'''
		This function finds the winning neuron
		'''
 		euclidDist=99999
 		BMUCoordinates=[]

 		for i in range(nKernel_y):
			kernelDist=self.getEuclideanDistance(self.map[i],inp)
			if kernelDist<euclidDist:
				euclidDist=kernelDist
				BMUCoordinates=i

		return BMUCoordinates

	def updateWeights(self,BMUCoordinates,currentVector):
		'''
		Updates the weight of the neurons of the SOM Map
		:params
		BMUCoordinates: Integer : Position of BMU Matrix
		currentVector: 3x300 Matrix containing current Input vector
		'''

		for i in range(nKernel_y):
			dist=self.getEuclideanDistance(BMUCoordinates,i)
			theta=np.exp(-(dist)**2/(2*(self.sigma)**2))
			self.map[i]+=(theta * self.learning_rate*(currentVector-self.map[i]))


	def distBetweenKernels(self,a,b):
		'''
		Returns the distance between 2 kernels
		:params
		a: Co-ordinates of winning neuron
		b: Co-ordinates of neuron with whom its distance is calculated
		'''
		dist=np.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)
		return dist

	def sigmaDecay(self,iterationCount):
		'''
		Returns the radius of the neighbours to update the weights
		i.e Decay function for sigma.
		'''
		self.timeConstant=self.num_iteration/np.log(self.sigma)
		self.sigma=self.sigma * np.exp(-iterationCount/float(self.timeConstant))

	def learningRateDecay(self,iterationCount):
		'''
		Decay function for Learning Rate
		'''
		self.learning_rate=self.learning_rate * np.exp(-iterationCount/float(self.num_iteration))

	def run(self):
		'''
		Primary starting function of the Self Organizing Maps Network
		Similar to fit function of sklearn library
		'''
		print "Running Self Organizing Map Neural Network"
		for i in range(self.num_iteration):
			print "Iteration #",i
			t1=time.time()
			self.sigmaDecay(i)
			self.learningRateDecay(i)
			for j in self.x_train:
				for k in range(len(j)-self.size_x):
					currentVector=j[k:k+self.size_x]
					BMUCoordinates=self.findBestMatchingUnit(currentVector)
					self.updateWeights(BMUCoordinates,currentVector)

			print "Iteration %d took: %d secs" % (i,time.time()-t1)

class CorrCoef_Layer2:
	def __init__(self,inp,som_kernels):
		'''
		Init for the 2nd layer of the Neural Network
		:params
		inp: Word2Vec for the input sentences
		som_kernels: Weights of the kernel learnt in the prevous layer
		'''

		self.input=inp
		self.map=weights

	def findCorrelationCoeff(self,a,b):
		'''
		Finds the correlation coefficient
		between two 2D Matrices
		:params
		a: Input(Word2Vec) ngram
		b: Learnt weights
		'''



if __name__=="__main__":
	som=SOM_Map(nKernel_y,size_y,size_x,learning_rate,sigma,num_iteration)
	som.run()
