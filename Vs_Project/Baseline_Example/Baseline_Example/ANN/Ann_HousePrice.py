from keras import layers,models
from keras import losses
from keras import optimizers
from keras import datasets
from sklearn import preprocessing
from matplotlib import pyplot as plt

from PlotWihtHistory import Plotting as pltting
from DataLib import DataGenerator 

class ANN(models.Model):

    def __init__(self,Nin,Nh,Nout):
        hidden = layers.Dense(Nh)
        output = layers.Dense(Nout)
        relu = layers.Activation('relu')

        x = layers.Input( shape=(Nin,))
        h = relu(hidden(x))
        y = output(h)

        opt = optimizers.Adam()

        super().__init__(x,y)
        self.compile(loss = 'mse' , optimizer = opt)



def main():
    Nin = 13
    Nh = 5
    Nout = 1

    model = ANN(Nin,Nh,Nout)

    (xTrain,yTrain),(xTest,yTest) = DataGenerator().Load_houseprice_normalized()


    history = model.fit(xTrain,yTrain , epochs = 100 , batch_size = 100 , validation_split = 0.2 , verbose = 1)

    pltting().plot_loss(history)
    pltting().plot_acc(history)

if __name__ == "__main__":
    main()