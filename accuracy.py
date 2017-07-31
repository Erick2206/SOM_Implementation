from minisom import MiniSom
from pprint import pprint
import numpy as np
from SOM import CorrCoef_Max_pooling_Layer2_3
from os.path import exists

map_x=10
map_y=10
size_x=3

class TrainCorrCoef:
    def __init__(self,corrCoef,map_y,map_x):
        self.corrCoef=corrCoef
        self.map_y=map_y
        self.map_x=map_x

    def train(self):
        som=MiniSom(self.map_y,self.map_x,54,sigma=1.0,learning_rate=0.5)
        som.random_weights_init(corrCoef)

        print("Training")
        som.train_random(corrCoef,100)

        print("\nReady!!!")

        return som


class Accuracy:
    def __init__(self,corrCoef,som,y_train,x_test,y_test):
        self.corrCoef=corrCoef
        self.som=som
        self.y_train=y_train
        self.x_test=x_test
        self.y_test=y_test
        self.map_dict=[[{0:0,1:0,'s':-1} for j in range(map_x) ]for i in range(map_y)]

    def findAccuracy(self):
        for cnt,xx in enumerate(self.corrCoef):
            w=som.winner(xx)
            self.map_dict[w[0]][w[1]][self.y_train[cnt]]+=1

        for q in range(map_y):
            for p in range(map_x):
                if self.map_dict[q][p][0]>self.map_dict[q][p][1]:
                    self.map_dict[q][p]['s']=0
                else:
                    self.map_dict[q][p]['s']=1

        if not exists('corrCoefListTest.npy'):
            trained_weights=np.load('trained_weights.npy')
            cc=CorrCoef_Max_pooling_Layer2_3(self.x_test,trained_weights,size_x)
            corrCoefListTest=cc.run()
            np.save('corrCoefListTest',corrCoefListTest)

        else:
            np.load('corrCoefListTest.npy')

        correct=0

        for cnt,xx in enumerate(corrCoefListTest):
            w=som.winner(xx)
            if self.map_dict[w[0]][w[1]]['s']==self.y_test[cnt]:
                correct+=1

        accuracy=float(correct)/len(y_test)

        print "Accuracy of the model is %.3f" % (accuracy)

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
    acc=Accuracy(corrCoef,som,y_train,x_test,y_test)
    acc.findAccuracy()
