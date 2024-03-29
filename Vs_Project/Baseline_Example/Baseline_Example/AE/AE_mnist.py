from keras import layers , models
from matplotlib import pyplot as plt

class AE(models.Model):
    def __init__(self,x_nodes,z_dim):
        x_shape = (x_nodes,)

        x = layers.Input(shape = x_shape)
        z = layers.Dense(z_dim,activation='relu')(x)
        y = layers.Dense(x_nodes , activation='sigmoid')(z)

        super().__init__(x,y)
        self.compile(optimizer = 'adadelta' , loss = 'binary_crossentropy',metrics=['accuracy'])

        self.x = x
        self.z = z
        self.z_dim = z_dim


    def Encoder(self):
        return models.Model(self.x , self.z)  # return new instance of model with original weight

    def Decoder(self):
        z_shape = (self.z_dim,)
        z = layers.Input(shape = z_shape)
        y_layer = self.layers[-1] # Get original layer and weight
        y = y_layer(z)
        return models.Model(z,y) # return new instance of model with original weight


from keras.datasets import mnist
import numpy as np
(X_train, _), (X_test, _) = mnist.load_data()

X_train = X_train.astype('float32') / 255.
X_test = X_test.astype('float32') / 255.
X_train = X_train.reshape((len(X_train), np.prod(X_train.shape[1:])))
X_test = X_test.reshape((len(X_test), np.prod(X_test.shape[1:])))
print(X_train.shape)
print(X_test.shape)

from PlotWihtHistory import Plotting as pltting


def show_ae(autoencoder):
    encoder = autoencoder.Encoder()
    decoder = autoencoder.Decoder()

    encoded_imgs = encoder.predict(X_test)
    decoded_imgs = decoder.predict(encoded_imgs)

    n = 10
    plt.figure(figsize=(20, 6))
    for i in range(n):

        ax = plt.subplot(3, n, i + 1)
        plt.imshow(X_test[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        ax = plt.subplot(3, n, i + 1 + n)
        plt.stem(encoded_imgs[i].reshape(-1))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        ax = plt.subplot(3, n, i + 1 + n + n)
        plt.imshow(decoded_imgs[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    plt.show()

def main():
    x_nodes = 784
    z_dim = 36

    autoencoder = AE(x_nodes, z_dim)

    history = autoencoder.fit(X_train, X_train,
                              epochs=10,
                              batch_size=256,
                              shuffle=True,
                              validation_data=(X_test, X_test))

    pltting().plot_acc(history, '(a) 학습 경과에 따른 정확도 변화 추이')
    plt.show()
    pltting().plot_loss(history, '(b) 학습 경과에 따른 손실값 변화 추이')
    plt.show()

    show_ae(autoencoder)
    plt.show()


if __name__ == '__main__':
    main()
    print()