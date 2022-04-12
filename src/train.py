from utils import setup_cuda_device
setup_cuda_device()

import os
import numpy as np
from absl import app
from tqdm import tqdm
from absl import flags
import tensorflow as tf
import matplotlib.pyplot as plt
#from predict import predict
from model import AutoEncoder
from data_manager import DataManager
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MSE

flags.DEFINE_integer("epochs", 1, "number of epochs")
flags.DEFINE_integer("batch_size", 2, "batch size")
flags.DEFINE_float("learning_rate", 0.005, "learning rate")
FLAGS = flags.FLAGS


def train(model):
    manager = DataManager()
    nbBatches = manager.size // FLAGS.batch_size
    lossTab = []
    loss = 0
    test = np.zeros((2, 41, 82, 1))

    for epoch in range(1,FLAGS.epochs+1):
        print('Epoch', epoch, '/', FLAGS.epochs)
        manager.shuffle()

        for i in tqdm(range(nbBatches)):
            (X1, X2), Y = manager.get_batch(FLAGS.batch_size, i)

            for x in range(manager.width):
                for y in range(manager.height):
                    R = np.concatenate((subImage(X1,79,x,y),subImage(X2,79,x,y)), axis=3)  # shape = (None,79,79,6)
                    #P = np.concatenate((subImage(X1,41,x,y),subImage(X2,41,x,y)), axis=2)  # shape = (None,41,82,3)
                    #pix = subImage(Y,1,x,y)                                                # shape = (None,1,1,3)
                    model.train_on_batch(R,test)

        lossTab.append(loss)
        print("Epoch {} - loss: {}".format(epoch, loss))

    print("Finished training.")

    # model.save("./trained_model/model.h5")
    # predict(model)

    plt.plot(lossTab)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()


def applyKernel(kernel, img):
    res = []
    for i in range(len(img)):
        res.append([[[0,0,0]]])

    return np.array(res)


def subImage(batch, size, x, y):
    sX = x-(size//2)
    maxX = len(batch[0])
    sY = y-(size//2)
    maxY = len(batch[0][0])
    res = []
    for i in range(batch.shape[0]):
        res.append([])
        for j in range(sX,sX+size):
            res[-1].append([])
            for k in range(sY,sY+size):
                res[-1][-1].append([0,0,0] if (j<0) | (j>=maxX) | (k<0) | (k>=maxY) else batch[i][j][k])
    return np.array(res)


def load_model(model):
    if os.path.isfile("./trained_model/model.h5"):
        print("Loading model from model.h5")
        model.load_weights("./trained_model/model.h5")
    else:
        print("model.h5 not found")


def main(argv):
    model = AutoEncoder()
    load_model(model)
    model.compile(loss=MSE, optimizer=Adam(FLAGS.learning_rate))
    model.summary()
    train(model)


if __name__ == '__main__':
    if not tf.test.is_gpu_available():
        print("WARNING: Not training with GPU. Training may be slow.")
    app.run(main)
