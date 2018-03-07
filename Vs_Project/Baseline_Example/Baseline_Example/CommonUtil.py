import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib
import datetime
from sklearn import metrics
from DataLib import DataGenerator_2D , Data2DMaker
from PlotWihtHistory import Plotting as pltting

def save_history_history(fname, history_history, fold=''):
    np.save(os.path.join(fold, fname), history_history)


def load_history_history(fname, fold=''):
    history_history = np.load(os.path.join(fold, fname)).item(0)
    return history_history

class Machinne_2D():
    def __init__(self, Model ,X,y,nb_classes = 2 , fig=True):
        self.nb_classes = nb_classes
        self.set_data(X,y)
        self.set_model(Model)
        self.fig = fig

    def set_data(self,X,y):
        nb_classes = self.nb_classes

        # Define Image Data using input dataset
        self.data = Data2DMaker(X,y,nb_classes)

    def set_model(self,Model):
        nb_classes = self.nb_classes
        data= self.data

        # Use CNN Class. Using other CNN is possible
        self.model = Model(in_shape = data.input_shape , nb_classes = nb_classes )

    def fit(self,nb_epoch = 10 , batch_size = 128 , verbose = 1):
        data = self.data
        model = self.model
        print(data.xTrain.shape)
        print(data.yTrain_c.shape)
        print(data.xTest.shape)
        print(data.yTest_c.shape)
        history = model.fit(data.xTrain, data.yTrain_c , 
                            batch_size = batch_size,
                            epochs = nb_epoch,
                            verbose = verbose ,
                            validation_data = (data.xTest , data.yTest_c))
        return history

    def run(self,nb_epoch = 30 , batch_size = 128 , verbose = 1):
        data = self.data
        model = self.model
        fig = self.fig

        history  = self.fit(nb_epoch , batch_size , verbose)
        score = model.evaluate(data.xTest , data.yTest_c , verbose = 1)

        print('Confusion Matrix')
        yTest_pred = model.predict(data.xTest, verbose = 1)
        yTest_pred = np.argmax(yTest_pred , axis = 1)
        print(metrics.confusion_matrix(data.yTest,yTest_pred))

        print('Test score:', score[0])
        print('Test accuracy:', score[1])

        #Save
        filename = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
        foldname = 'output_' + filename
        os.makedirs(foldname)
        save_history_history(
            'history_history.npy', history.history, fold=foldname)
        model.save_weights(os.path.join(foldname, 'dl_model.h5'))
        print('Output results are saved in', foldname)
        if fig:
            plt.figure(figsize=(12, 4))
            plt.subplot(1, 2, 1)
            pltting().plot_acc(history)
            plt.subplot(1, 2, 2)
            pltting().plot_loss(history)
            plt.show()