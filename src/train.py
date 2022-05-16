from utils import setup_cuda_device
setup_cuda_device("0")

import os
import numpy as np
from absl import app
from tqdm import tqdm
from absl import flags
import tensorflow as tf
import matplotlib.pyplot as plt
from predict import predict
from random import randint
from model import AutoEncoder
from data_manager import DataManager
from tensorflow.keras.optimizers import Adamax
from tensorflow.keras.losses import MSE
from tensorflow.keras.losses import MAE

flags.DEFINE_integer("epochs", 1, "number of epochs")
flags.DEFINE_integer("batch_size", 32, "batch size")
flags.DEFINE_integer("nbPixelsPick", 128, "number of pixel pick by batch")  # use 128 in paper
flags.DEFINE_float("learning_rate", 0.001, "learning rate")                 # use 0.001 in paper
FLAGS = flags.FLAGS


def train(model):
    manager = DataManager(frameSkip=720)
    nbBatches = manager.nbBatches // FLAGS.batch_size
    lossTab = []
    loss = 0

    for epoch in range(1,FLAGS.epochs+1):
        manager.shuffle()

        for i in tqdm(range(nbBatches),desc="Epoch "+str(epoch)+"/"+str(FLAGS.epochs)):
            (X1, X2), Y = manager.get_batch(FLAGS.batch_size, i)
            X1X2 = np.concatenate((X1,X2), axis=4)

            for _ in range(FLAGS.nbPixelsPick):
                x = randint(0,manager.height-1)
                y = randint(0,manager.width-1)
                loss = model.train_on_batch(X1X2[:, x:x+79, y:y+79, :, :], np.squeeze(Y[:, x+39:x+40, y+39:y+40, :, :]))

        lossTab.append(loss)
        print("Epoch {} - loss: {}".format(epoch, loss))

    print("Finished training.")

    model.save("../trained_model/model.h5")
    predict(model, video=False,frameSkip=780)

    plt.plot(lossTab)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()


def load_model(model):
    if os.path.isfile("../trained_model/model.h5"):
        print("Loading model from model.h5")
        model.load_weights("../trained_model/model.h5")
    else:
        print("model.h5 not found")


def main(argv):
    model = AutoEncoder()
    load_model(model)
    model.summary()
    model.compile(loss=MSE, optimizer=Adamax(FLAGS.learning_rate))
    #model.compile(loss=MAE, optimizer=Adamax(FLAGS.learning_rate))
    train(model)


if __name__ == '__main__':
    if not tf.test.is_gpu_available():
        print("WARNING: Not training with GPU. Training may be slow.")
    app.run(main)
