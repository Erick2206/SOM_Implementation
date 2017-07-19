from minisom import MiniSom
from numpy import genfromtxt,array,linalg,zeros,mean,std,apply_along_axis,load

data=load('corrCoef.npy')

som=MiniSom(10,10,54,sigma=1.0,learning_rate=0.5)
som.random_weights_init(data)

print("Training")
som.train_random(data,100)

print("\nReady!!!")

from matplotlib.pyplot import plot,axis,show,pcolor,colorbar,bone

bone()
pcolor(som.distance_map().T)
colorbar()

target=load('input.npy')[1]

markers=['o','s']
colors=['r','g']

for cnt,xx in enumerate(data):
	w=som.winner(xx)
	plot(w[0]+.5,w[1]+.5, markers[target[cnt]],markerfacecolor='None',markeredgecolor=colors[target[cnt]],markersize=12,markeredgewidth=2)

axis([0,10,0,10])

show()
