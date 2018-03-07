import numpy as np
import pickle
import os

img_size = 32
num_channels = 3
img_size_flat = img_size * img_size * num_channels
num_classes = 10

_num_files_train = 5
_images_per_file = 10000
_num_images_train = _num_files_train * _images_per_file


def _unpickle(filename):
    with open(filename, mode='rb') as file:
        data = pickle.load(file, encoding='bytes')
    return data

def _convert_images(raw):
    raw_float = np.array(raw, dtype=float) / 255.0
    images = raw_float.reshape([-1, num_channels, img_size, img_size])
    images = images.transpose([0, 2, 3, 1])
    return images

def _load_data(filename):
    data = _unpickle(filename)
    raw_images = data[b'data']
    cls = np.array(data[b'labels'])
    images = _convert_images(raw_images)
    return images, cls

def load_class_names(basepath):
    raw = _unpickle(filename= os.path.join(basepath,"batches.meta"))[b'label_names']
    names = [x.decode('utf-8') for x in raw]
    return names

def load_training_data(basepath):
    images = np.zeros(shape=[_num_images_train, img_size, img_size, num_channels], dtype=float)
    cls = np.zeros(shape=[_num_images_train], dtype=int)
    begin = 0
    for i in range(_num_files_train):
        images_batch, cls_batch = _load_data(filename=os.path.join(basepath,"data_batch_") + str(i + 1))
        num_images = len(images_batch)
        end = begin + num_images
        images[begin:end, :] = images_batch
        cls[begin:end] = cls_batch
        begin = end
    return images, cls

def load_test_data(basepath):
    images, cls = _load_data(filename=os.path.join(basepath,"test_batch"))
    return images, cls

localpath = r"F:\kaggle_data\cifar-10-python\cifar-10-batches-py"
def Load_CIFAR10(basepath = localpath):
    xTrain,yTrain = load_training_data(basepath)
    xTest,yTest = load_test_data(basepath)
    return (xTrain,yTrain),(xTest,yTest)


if __name__=="__main__":
    (x,y),(x0,y0) = Load_CIFAR10(localpath)
    print(x.shape)
