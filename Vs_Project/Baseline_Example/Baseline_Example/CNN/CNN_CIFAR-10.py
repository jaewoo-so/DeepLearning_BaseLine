import keras
from keras import models , datasets
from keras.models import Model
from keras import backend
from keras import losses , optimizers
from keras.layers import Dense , Conv2D , MaxPool2D , Dropout , Flatten , Input
import matplotlib.pyplot as plt
import numpy as np
import os
import datetime
import CommonUtil

from PlotWihtHistory import Plotting as pltting
from DataLib import DataGenerator_2D , Data2DMaker
import LocalDataLoader 
from CommonUtil import Machinne_2D

class CNN(models.Model):
    def __init__(self, in_shape , nb_classes):
        self.nb_class = nb_classes
        self.in_shape = in_shape
        self.build_model()
        super().__init__(self.x ,self.y)
        self.compile(loss = 'categorical_crossentropy' , optimizer = 'adadelta' , metrics=['accuracy'])

    def build_model(self):
        nb_classes = self.nb_class
        in_shape = self.in_shape

        x = Input(in_shape)
        h = Conv2D(32 , kernel_size = (3,3) , activation = 'relu' , input_shape = in_shape)(x)
        h = Conv2D(64,(3,3),activation='relu')(h)
        h = MaxPool2D((2,2))(h)
        h = Dropout(0.25)(h)
        h = Flatten()(h)

        z_cl = h

        h = Dense(128 , activation='relu')(h)
        h = Dropout(0.5)(h)

        z_fl = h

        y = Dense(nb_classes , activation='softmax' , name='preds')(h)

        self.cl_part = Model(x,z_cl)
        self.fl_part = Model(x,z_fl)
        self.x = x
        self.y = y

class CNN_CIFAR10_Machine(Machinne_2D):
    def __init__(self):
        (xTrain,yTrain),(xTest,yTest) = LocalDataLoader.Load_CIFAR10()
        super().__init__(CNN,xTrain,yTrain, nb_classes = 10 )

def main():
    Runner = CNN_CIFAR10_Machine()
    Runner.run(nb_epoch = 2)

if __name__ == "__main__":
    main()