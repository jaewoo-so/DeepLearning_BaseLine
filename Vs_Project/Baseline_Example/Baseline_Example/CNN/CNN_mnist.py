import keras
from keras import models , layers
from keras import backend
from keras import losses , optimizers
from PlotWihtHistory import Plotting as pltting
from DataLib import DataGenerator_2D

class CNN(models.Sequential):
    def __init__(self, input_shape , num_classes):
        super().__init__()
        self.add(layers.Conv2D(32 , kernel_size = (3,3) , activation = 'relu' , input_shape = input_shape))
        self.add(layers.Conv2D(64, (3, 3), activation='relu'))
        self.add(layers.MaxPooling2D(pool_size = (2,2)))
        self.add(layers.Dropout(0.25))
        self.add(layers.Flatten())
        self.add(layers.Dense(128,activation = 'relu'))
        self.add(layers.Dropout(0.5))
        self.add(layers.Dense(num_classes , activation = 'softmax'))

        self.compile(loss = losses.categorical_crossentropy , optimizer = optimizers.Adam() , metrics = ['accuracy'])


def main():
    data = DataGenerator_2D().Load_mnist()
    
    print(data.xTrain.shape)
    print(data.xTest.shape)
    model = CNN(data.input_shape , data.num_classes)

    history = model.fit(data.xTrain,data.yTrain , batch_size = 100 , epochs = 20 , validation_split = 0.3 , verbose = 1)

    acc = model.evaluate(data.xTest , data.yTest ,verbose = 1)

    print("Acc : {0}",acc)

    pltting().plot_acc(history)
    pltting().plot_loss(history)

if __name__ == "__main__":
    main()