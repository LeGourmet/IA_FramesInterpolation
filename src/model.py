import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Conv2D, Reshape, BatchNormalization, Concatenate


def Conv(n_filters, filter_width, strides, activation="relu"):
    return Conv2D(n_filters, filter_width, strides=strides, use_bias=False, padding="valid", activation=activation)


def AutoEncoder():
    input = Input(shape=(79, 79, 6), name="Input")

    X = Conv(32, 7, 1)(input)
    #X = BatchNormalization()(X)
    X = Conv(32, 2, 2)(X)

    X = Conv(64, 5, 1)(X)
    #X = BatchNormalization()(X)
    X = Conv(64, 2, 2)(X)

    X = Conv(128, 5, 1)(X)
    #X = BatchNormalization()(X)
    X = Conv(128, 2, 2)(X)

    X = Conv(256, 3, 1)(X)
    #X = BatchNormalization()(X)
    X = Conv(2048, 4, 1)(X)
    X = Conv(3362, 1, 1, activation="softmax")(X)

    kernel = Reshape((41, 82, 1), name="kernel")(X)

    R1, R2 = tf.split(input, 2, axis=3)
    P = Concatenate(axis=2)([R1[:, 19:60, 19:60, :],R2[:, 19:60, 19:60, :]])
    kernel3 = tf.repeat(kernel, repeats=3, axis=3) # need to be None,41,82,3 => 3x same

    output = tf.math.reduce_sum(tf.math.multiply(kernel3,P),[1,2], name="output")

    return Model(inputs=input, outputs=output, name="model")

