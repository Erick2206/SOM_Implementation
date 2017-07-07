'''
This file is an implementation of the SOM algorithm
'''

import numpy as np
from pprint import pprint

#Kernel Parameters
size_x=8			#size of 1 kernel
size_y=8

#SOM Map parameters
nKernel_x=8			#No of kernels on x and y axis on a SOM Map.
nKernel_y=8

#Training parameters
learning_rate=0.1
sigma=0.3
inp=np.random.rand(size_x,size_y)


class SOM_Map:
	def __init__(self,nKernel_y,nKernel_x,size_y,size_x,learning_rate=0.1,sigma=0.3,num_iteration=10000):
		self.nKernel_y=nKernel_y
		self.nKernel_x=nKernel_x
		self.size_y=size_y
		self.size_x=size_x
		self.learning_rate=learning_rate
		self.sigma=sigma
		self.num_iteration=num_iteration

		self.map=self.setRandomWeights()

	def self.setRandomWeights():
		'''
		Returns a map of Randomly initiated weights
		'''
		kernels=[[self.get_kernel(size_y,size_x) for j in range(nKernel_y)] for i in range(nKernel_x)]
		return kernels

	def self.get_kernel(size_y,size_x):
		'''
		Returns a kernel of size_y rows and size_x columns
		'''
		kernel=np.random.rand(size_x,size_y)
		return kernel

	def self.getDistance(a,b):
		'''
		Returns the Euclidean Distance between 2 Matrices
		'''
		total=0
		for i in range(self.size_y):
			for j in range(self.size_x):
				total+=(a[i][j]-b[i][j])**2

		return np.sqrt(total)

	def self.fit():
		D=self.getDistance(a,b)

	def self.updateWeights():
		#TODO

	def self.getNeighbours():
		#TODO

if __name__=="__main__":
	som=SOM_Map(nKernel_x,nKernel_y,size_y,size_x,learning_rate,sigma)
	pprint(som.map)
