

import tensorflow, keras
print(tensorflow.__version__, keras.__version__)

from keras.models import Sequential
from keras.layers import LeakyReLU
from keras.layers import InputLayer

m = Sequential()
m.add(InputLayer(input_shape=(1000,)))
m.add(LeakyReLU(alpha=0.2))