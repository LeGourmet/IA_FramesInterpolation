import tensorflow as tf
import numpy as np
from tensorflow.keras import Input, Sequential
from tensorflow.keras.layers import Conv2D, Reshape, BatchNormalization


def Conv(n_filters, filter_width, strides, activation="relu"):
    return Conv2D(n_filters, filter_width, strides=strides, use_bias=False, padding="valid", activation=activation)


def buildModel():
    model = Sequential()
    model.add(Input(shape=(79, 79, 6), name="Input"))

    model.add(Conv(32, 7, 1))
    # batch normalize
    model.add(Conv(32, 2, 2))

    model.add(Conv(64, 5, 1))
    # batch normalize
    model.add(Conv(64, 2, 2))

    model.add(Conv(128, 5, 1))
    # batch normalize
    model.add(Conv(128, 2, 2))

    model.add(Conv(256, 3, 1))
    # batch normalize
    model.add(Conv(2048, 4, 1))
    model.add(Conv(3362, 1, 1, activation="softmax"))  # ou activation="linear"

    model.add(Reshape((41, 82), name="Output"))
    return model


def myLossFnct(pixel= np.array([1,2,3])):
    def loss(kernel, patchs):
        print(kernel.shape, patchs.shape, pixel.shape)
        return tf.constant(1.)
    return loss
