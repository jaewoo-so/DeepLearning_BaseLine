from PIL import Image
import numpy as np
import math
import os
from keras import backend as K
from keras.layers import *
from keras import models , optimizers 
import tensorflow as tf
import argparse
from LocalDataLoader import Load_CIFAR10

K.set_image_data_format("channels_first")

def mse_4d(y_true , y_pred):
    return K.mean(K.square( y_pred  - y_true), axis = (1,2,3))

def mse_4d(y_true , y_pred):
    return tf.reduce_mean( tf.square(y_pred - y_true) , axis = (1,2,3))

class GAN_CNN(models.Sequential):
    def __init__(self, input_dim=64):
        super.__init__()
        self.input_dim = input_dim
        self.generator = self.GENERATOR()
        self.discriminator = self.DISCRIMINATOR()

        self.add(self.generator)
        self.discriminator.trainable = False
        self.add(self.discriminator)
        self.compile_all()

    def compile_all(self):
        d_optim = optimizers.SGD(lr = 0.0005 , momentum = 0.9 , nesterov = True)
        g_optim = optimizers.SGD(lr = 0.0005 , momentum = 0.9 , nesterov = True)

        self.generator.compile(loss = mse_4d , optimizers = g_optim)
        self.compile(loss = "binary_crossentropy" , optimizer = g_optim)
        self.discriminator.trainable = True
        self.discriminator.compile(loss = mse_4d , optimizers = d_optim)

    def GENERATOR(self):

        input_dim = self.input_dim

        model = models.Sequential()
        model.add(layers.Dense(1024, activation='tanh', input_dim=input_dim))
        model.add(layers.Dense(128 * 7 * 7, activation='tanh'))
        model.add(layers.BatchNormalization())
        model.add(layers.Reshape((128, 7, 7), input_shape=(128 * 7 * 7,)))
        model.add(layers.UpSampling2D(size=(2, 2))) # 14x14 image
        model.add(layers.Conv2D(64, (5, 5), padding='same', activation='tanh'))
        model.add(layers.UpSampling2D(size=(2, 2))) # 28x28 image
        model.add(layers.Conv2D(1, (5, 5), padding='same', activation='tanh'))
        return model

    def DISCRIMINATOR(self):
        model = models.Sequential()
        model.add(layers.Conv2D(64, (5, 5), padding='same', activation='tanh',input_shape=(1, 28, 28)))
        model.add(layers.MaxPooling2D(pool_size=(2, 2))) # 14x14 image
        model.add(layers.Conv2D(128, (5, 5), activation='tanh'))
        model.add(layers.MaxPooling2D(pool_size=(2, 2))) # 7x7 image
        model.add(layers.Flatten())
        model.add(layers.Dense(1024, activation='tanh'))
        model.add(layers.Dense(1, activation='sigmoid'))
        return model

    def get_z(self, ln):
        input_dim = self.input_dim
        return np.random.uniform(-1, 1, (ln, input_dim))

    def train_both(self,x):
        ln = x.shape[0]
        z = self.get_z(ln)

        fake = self.generator.predict(z)

        real_fake_img = np.concatenate(x,fake)
        real_fake_label = [1]*ln + [0]*ln
        
        d_loss = self.discriminator.train_on_batch(real_fake_img , real_fake_label)

        z = sepf;get_z(ln)
        self.discriminator.trainable = False
        g_loss = self.train_on_batch(z , [1]*ln)
        self.discriminator.trainable = True

        return d_loss , g_loss

## -- modeule -- ##
def get_x(xTrain,index,batch_size):
    return xTrain[index*batch_size : (index+1 )*batch_size]

def combine_images(generated_images):
    num = generated_images.shape[0]
    width = int(math.sqrt(num))
    height = int(math.ceil(float(num) / width))
    shape = generated_images.shape[2:]
    image = np.zeros((height * shape[0], width * shape[1]),
                     dtype=generated_images.dtype)
    for index, img in enumerate(generated_images):
        i = int(index / width)
        j = index % width
        image[i * shape[0]:(i + 1) * shape[0],
        j * shape[1]:(j + 1) * shape[1]] = img[0, :, :]
    return image

def save_images(generated_images, output_fold, epoch, index):
    image = combine_images(generated_images)
    image = image * 127.5 + 127.5
    Image.fromarray(image.astype(np.uint8)).save(
        output_fold + '/' +
        str(epoch) + "_" + str(index) + ".png")

def train(args):
    BATCH_SIZE = args.batch_size
    epochs = args.epochs
    output_fold = args.output_fold
    input_dim = args.input_dim
    n_train = args.n_train

    os.makedirs(output_fold , exist_ok=True)
    print('Output_fold is', output_fold)

    (xTrain , _) , (_,_) = Load_CIFAR10()
    xTrain = (xTrain.astype(np.float32) - 127.5)/127.5
    xTrain  = xTrain.reshape( (xTrain.shape[0] , 1) + xTrain.shape[1:] )
    gan = GAN_CNN(input_dim)

    d_loss_ll = []
    g_loss_ll = []

    # 1epoch do all batch data
    for epoch in range(epochs):
        print("Epoch is" , epoch)
        print("Num of Batches" , int(xTrain.shape[0]/BATCH_SIZE))

        d_loss_l = []
        g_loss_l = []

        for index in range(int(xTrain.shape[0] / BATCH_SIZE)):
            x = get_x(xTrain, index , BATCH_SIZE)

            # First Train discriminator with [real,fake] -> update discriminator
            # Second Train generator with D(G[fake]) -> update generator only, discriminatro trainable = false
            d_loss , g_loss = gan.train_both(x)

            d_loss_l.append(d_loss)
            g_loss_l.append(g_loss)

        if epoch % 10 == 0 or epeoch == epochs - 1:
            print("Save Fake Image")
            z = gan.get_z(x.shape[0])
            fake_pred = gan.generator(z , verbose  = 0)
            save_Image


def main():
    parser = argparse.ArgumentParser()