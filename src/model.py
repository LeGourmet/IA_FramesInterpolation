from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Conv2D, Reshape


def Conv(n_filters, filter_width, strides, activation="relu"):
    return Conv2D(n_filters, filter_width, strides=strides, use_bias=False, padding="valid", activation=activation)


def AutoEncoder():
    input = Input(shape=(79, 79, 6), name="Input")

    X = Conv(32, 7, 1)(input)
    # batch normalize
    X = Conv(32, 2, 2)(X)

    X = Conv(64, 5, 1)(X)
    # batch normalize
    X = Conv(64, 2, 2)(X)

    X = Conv(128, 5, 1)(X)
    # batch normalize
    X = Conv(128, 2, 2)(X)

    X = Conv(256, 3, 1)(X)
    # batch normalize
    X = Conv(2048, 4, 1)(X)
    X = Conv(3362, 1, 1, activation="softmax")(X) # ou activation="linear"

    output = Reshape((41, 82), name="Output")(X)
    return Model(inputs=input, outputs=output, name="model")

