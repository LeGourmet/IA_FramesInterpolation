import os
import matplotlib.pyplot as plt
from model import buildModel


def sample(model):
    X = model.predict()
    plt.imshow()
    plt.show()


if __name__ == '__main__':
    model = buildModel()

    if os.path.isfile("./trained_model/model.h5"):
        print("Loading model from model.h5")
        model.load_weights("./trained_model/model.h5")
    else:
        print("model.h5 not found")

    sample(model)
