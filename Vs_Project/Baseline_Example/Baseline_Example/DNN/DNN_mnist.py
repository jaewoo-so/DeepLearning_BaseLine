import keras 
from keras import models, layers,losses , optimizers 

from DataLib import DataGenerator as dataGen
from PlotWihtHistory import Plotting as pltting


class DNN_mnist(models.Sequential):
    def __init__(self,Nin,Nhs,Nout):
        super().__init__()
        self.add(layers.Dense(Nhs[0], activation = 'relu' , input_shape = (Nin,) , name="Hidden_1"))
        self.add(layers.Dropout(0.2))
        self.add(layers.Dense(Nhs[1] , activation = 'relu' , name = "Hidden_2"))
        self.add(layers.Dropout(0.2))
        self.add(layers.Dense(Nout , activation = 'softmax' ))

        self.compile(loss = losses.categorical_crossentropy , optimizer = optimizers.Adam() , metrics = ['accuracy'])


def main():
    Nin  = 784 
    Nout = 10
    Nhs = [100,50]

    (xTrain,yTrain) , (xTest , yTest) = dataGen().Load_mnist()


    model = DNN_mnist(Nin,Nhs,Nout)

    history = model.fit(xTrain,yTrain , epochs = 10 , validation_split=0.2 , batch_size = 100,verbose=1)

    acc = model.evaluate(xTest,yTest)
    
    print("Accuracy : " , acc)

    pltting().plot_loss(history)
    pltting().plot_acc(history)

    print()


if __name__ == "__main__":
    main()

