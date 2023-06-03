import tensorflow as tf

from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Conv2D, Reshape, BatchNormalization, Activation


def Conv(n_filters, filter_width, strides, activation="relu"):
    return Conv2D(n_filters, filter_width, strides=strides, use_bias=False, padding="valid", activation=activation)


def AutoEncoder():
    input = Input(shape=(79, 79, 3, 2), name="Input")

    X = Reshape((79, 79, 6))(input)
    X = Conv(32, 7, 1)(X)
    X = BatchNormalization()(X)
    X = Conv(32, 2, 2)(X)

    X = Conv(64, 5, 1)(X)
    X = BatchNormalization()(X)
    X = Conv(64, 2, 2)(X)

    X = Conv(128, 5, 1)(X)
    X = BatchNormalization()(X)
    X = Conv(128, 2, 2)(X)

    X = Conv(256, 3, 1)(X)
    X = BatchNormalization()(X)
    X = Conv(2048, 4, 1)(X)
    X = Conv(3362, 1, 1)(X)
    X = Activation("softmax")(X)  # softamx because we are dealing with a convolution kernel

    # convolution of the kernel inside the model for derivation reasons
    kernel = Reshape((41, 41, 1, 2))(X)
    output = tf.math.reduce_sum(tf.math.multiply(tf.repeat(kernel, repeats=3, axis=3), input[:, 19:60, 19:60, :, :]), [1, 2, 4])

    return Model(inputs=input, outputs=output, name="model")
