from minisom import MiniSom
from pprint import pprint
import numpy as np

map_x=10
map_y=10

pprint (map_dict)
print len(map_dict[0])
'''
for cnt,xx in enumerate(x_train):
    w=som.winner(xx)
    print w
    break
'''

class TrainCorrCoef:
    def __init__(self,corrCoef,map_y,map_x):
        self.corrCoef=corrCoef
        self.map_y=map_y
        self.map_x=map_x

    def train(self):
        som=MiniSom(self.map_y,self.map_x,54,sigma=1.0,learning_rate=0.5)
        som.random_weights_init(data)

        print("Training")
        som.train_random(data,100)

        print("\nReady!!!")

        return som

class Accuracy:
    def __init__(self,som,x_test,y_test):
        self.som=som
        self.x_test=x_test
        self.y_test=y_test
        self.map_dict=[[{0:0,1:0,'s':-1} for j in range(map_x) ]for i in range(map_y)]

    def findAccuracy(self):

        print "Accuracy of the model is",

if __name__=='__main__':
    '''
    Train max pooling of Correlation coefficients using MiniSom
    '''
    corrCoef=np.load('corrCoef.npy')
    corrCoefTrain=TrainCorrCoef(corrCoef,map_y,map_x)
    som=corrCoefTrain.train()

    '''
    Find the accuracy of the test data
    '''
    #Load data
    x_train, y_train, x_test, y_test, vocabulary_inv= np.load('input.npy')
    acc=Accuracy(som,x_test,y_test)
    acc.findAccuracy()
