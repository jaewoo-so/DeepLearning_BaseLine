import numpy as np
from keras import datasets  # mnist
from keras.utils import np_utils  # to_categorical
from sklearn import preprocessing 
from keras import utils as kUtil
from keras import backend as K

class DataGenerator():
    def Load_mnist(self):
        (xTrain, yTrain), (xTest, yTest) = datasets.mnist.load_data()
    
        yTrain = np_utils.to_categorical(yTrain)
        yTest = np_utils.to_categorical(yTest)
    
        L, W, H = xTrain.shape
        xTrain = xTrain.reshape(-1, W * H)
        xTest = xTest.reshape(-1, W * H)
    
        xTrain = xTrain / 255.0
        xTest = xTest / 255.0
    
        return (xTrain, yTrain), (xTest, yTest)
    
    def Load_houseprice_normalized(self):
        (xTrain, yTrain), (xTest, yTest) = datasets.boston_housing.load_data()
        scaler = preprocessing.MinMaxScaler()
        xTrain = scaler.fit_transform(xTrain)
        xTest = scaler.transform(xTest)
        return (xTrain, yTrain), (xTest, yTest)




class DataGenerator_2D():
    def add_channels(self):
        X = self.X

        if len(X.shape) == 3: # if Gray Scale (1 channel), X.Shape = (N,row,col). need to convert (Xmrow,col,1)
            N , img_rows , img_cols = X.shape
            X = X.reshape(X.shape[0],img_rows , img_cols ,1 )
            input_shape = (img_rows,img_cols,1)
        else:
            input_shape = X.shape[1:]
        self.X = X
        self.input_shape = input_shape
            

    def Load_mnist(self):
        num_classes = 10

        (xTrain, yTrain), (xTest, yTest) = datasets.mnist.load_data()

        trainNum , imgH , imgW = xTrain.shape
        testNum = xTest.shape[0]
        input_shape = None
        if K.image_data_format() == 'channels_first':
            xTrain = xTrain.reshape(trainNum ,1 , imgH , imgW )
            xTest = xTest.reshape(testNum , 1 , imgH , imgW)
            input_shape = (1,imgH,imgW)
        else :
            xTrain = xTrain.reshape(trainNum , imgH , imgW , 1)
            xTest = xTest.reshape(testNum ,imgH , imgW , 1)
            input_shape = (imgH,imgW,1)

        xTrain = xTrain.astype('float32')
        xTrain /=255.0
        xTest = xTest.astype('float32')
        xTest /= 255.0

        yTrain = kUtil.to_categorical(yTrain , num_classes)
        yTest = kUtil.to_categorical(yTest , num_classes)

        #self.input_shape = input_shape
        #self.num_classes = num_classes
        #self.xTrain , self.yTrain = xTrain , yTrain
        #self.xTest , self.yTest = xTest , yTest
        return CatergoricalDataInfo_splited(input_shape , num_classes , xTrain , yTrain , xTest , yTest)

    def Load_CIFAR10(self):
        (xTrain, yTrain), (xTest, yTest) = datasets.boston_housing.load_data()


class CatergoricalDataInfo_splited():
    def __init__(self, input_shape , num_classes , xTrain,yTrain,xTest,yTest):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.xTrain = xTrain
        self.yTrain = yTrain
        self.xTest = xTest
        self.yTest = yTest