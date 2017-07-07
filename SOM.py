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


class SOM_Map:
	def __init__(self,nKernel_y,nKernel_x,size_y,size_x):
		self.map=[[self.get_kernel(size_y,size_x) for j in range(nKernel_y)] for i in range(nKernel_x)]
		s

	def get_kernel(size_y,size_x):
		'''
		Returns a kernel of size_y rows and size_x columns
		'''
		kernel=np.random.rand(size_x,size_y)
		return kernel

	def self.getDistance():
		

	def self.fit():
		D=self.getDistance(a,b)

	def self.updateWeights():
		#TODO

	def self.getNeighbours():
		#TODO

if __name__=="__main__":
	som=SOM_Map(nKernel_x,nKernel_y,size_y,size_x)
	pprint(som.map)
