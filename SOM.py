'''
This file is an implementation of the SOM algorithm
'''

import numpy as np

#Kernel Parameters
size_x=8			#size of 1 kernel
size_y=8

#SOM Map parameters
nKernel_x=8			#No of kernels on x and y axis on a SOM Map.
nKernel_y=8

class kernel:
	def __init__(self, size_y, size_x):
		#Randomly initialized kernel
		self.kernel=np.random.rand(size_x,size_y)
		
class SOM_Map:
	def __init__(self,nKernel_y,nKernel_x):



if __name__=="__main__":
	k=kernel(size_y,size_x)