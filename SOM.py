'''
This file is an implementation of the SOM algorithm
'''

import numpy as np
from pprint import pprint
from sentiment_som import load_data
from w2v import train_word2vec

#Kernel Parameters
size_x=3			#size of 1 kernel
size_y=300

#SOM Map parameters
nKernel_y=100 		#No of kernels on x and y axis on a SOM Map.

#Training parameters
learning_rate=0.1
sigma=0.3

#Word2Vec parameters
embedding_dim=300
min_word_count=1
context=10

#Convolution parameters
ngram=3

#Loads the dataset matrices, the Word2Vec weights and the vocabulary
x_train, y_train, x_test, y_test, vocabulary_inv = load_data()
embedding_weights = train_word2vec(np.vstack((x_train, x_test)), vocabulary_inv, num_features=embedding_dim,
								   min_word_count=min_word_count, context=context)

print len(x_train)


class SOM_Map:
	def __init__(self,nKernel_y,nKernel_x,size_y,size_x,learning_rate=0.1,sigma=0.3,num_iteration=10000):
		self.nKernel_y=nKernel_y
		self.nKernel_x=nKernel_x
		self.size_y=size_y
		self.size_x=size_x
		self.learning_rate=learning_rate
		self.sigma=sigma

		#Total no. of iterations for the model
		self.num_iteration=num_iteration

		#Iteration count for finding the value of sigma
		self.iterationCount=1

		self.map=self.setRandomWeights()

	def setRandomWeights(self):
		'''
		Returns a map of Randomly initiated weights
		'''
		kernels=[[self.get_kernel(size_y,size_x) for j in range(nKernel_y)] for i in range(nKernel_x)]
		return kernels

	def get_kernel(self,size_y,size_x):
		'''
		Returns a kernel of size_y rows and size_x columns
		'''
		kernel=np.random.random_sample((self.size_y,self.size_x))
		return kernel

	def getEuclideanDistance(self,a,b):
		'''
		Returns the Euclidean Distance between 2 Matrices
		'''
		total=0
		for i in range(self.size_y):
			for j in range(self.size_x):
				total+=(a[i][j]-b[i][j])**2

		return np.sqrt(total)

	def findBestMatchingUnit(self):
		'''
		This function finds the winning neuron
		'''
 		euclidDist=99999
 		BMUCoordinates=[]

 		for i in range(nKernel_y):
 			for j in range(nKernel_x):
 				kernelDist=self.getEuclideanDistance(self.map[i][j],inp) #Need work here because there is no input dataset at present

 				if kernelDist<euclidDist:
 					euclidDist=kernelDist
 					BMUCoordinates=[i,j]

 		return BMUCoordinates

	def updateWeights(self):
		'''
		Updates the weight of the neurons of the SOM Map
		'''

		for i in range(nKernel_y):
			for j in range(nKernel_x):
				dist=self.distanceBetweenKernels(BMU,[i,j])
				eta=np.exp(-(dist)**2/(2*(self.sigma)**2))
				self.learningRateDecay()
				self.map[i][j]+=(eta*self.learning_rate*(inp-self.map[i][j]))
		return eta

	def getNeighbourRadius(self):
		'''
		Returns the radius of the neighbours to update the weights
		'''
		self.sigma=max(nKernel_x,nKernel_y)/2
		self.timeConstant=num_iteration/np.log(self.sigma)
		self.sigma=self.sigma * np.exp(-self.iterationCount/float(self.timeConstant))
		self.iterationCount+=1
		return self.sigma

	def learningRateDecay(self):
		'''
		Decay function for Learning Rate
		'''
		self.learning_rate=self.learning_rate * np.exp(-self.iterationCount/float(self.timeConstant))

	def distBetweenKernels(self,a,b):
		'''
		Returns the distance between 2 kernels
		:params
		a: Co-ordinates of winning neuron
		b: Co-ordinates of neuron with whom its distance is calculated
		'''
		dist=np.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)
		return dist

	def fit(self):
		'''
		Primary starting function of the Self Organizing Maps Network
		Similar to fit function of sklearn library
		'''
		for i in range(self.num_iteration):
			for j in range(len(x_train)):
				BMU=self.findBestMatchingUnit()
				self.updateWeights(BMU)



if __name__=="__main__":
	som=SOM_Map(nKernel_x,nKernel_y,size_y,size_x,learning_rate,sigma)
	pprint(som.map)
