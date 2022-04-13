from utils import setup_cuda_device, subImage
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
flags.DEFINE_integer("batch_size", 32, "batch size")
flags.DEFINE_float("learning_rate", 0.005, "learning rate")
FLAGS = flags.FLAGS


def train(model):
    manager = DataManager()
    nbBatches = manager.size // FLAGS.batch_size
    lossTab = []
    loss = 0

    for epoch in range(1,FLAGS.epochs+1):
        print('Epoch', epoch, '/', FLAGS.epochs)
        manager.shuffle()

        for i in tqdm(range(nbBatches)):
            (X1, X2), Y = manager.get_batch(FLAGS.batch_size, i)

            for x in range(manager.width):
                for y in range(manager.height):
                    R = np.concatenate((subImage(X1,79,x,y),subImage(X2,79,x,y)), axis=3)  # shape = (None,79,79,6)
                    model.train_on_batch(R, Y[:,x:x+1,y:y+1,:]) # pre calcul pix ? => #shape none,1,1,3 to none,3

        lossTab.append(loss)
        print("Epoch {} - loss: {}".format(epoch, loss))

    print("Finished training.")

    # model.save("./trained_model/model.h5")
    # predict(model)

    plt.plot(lossTab)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()


def custom_loss(y_true,y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred), axis=-1)


def load_model(model):
    if os.path.isfile("./trained_model/model.h5"):
        print("Loading model from model.h5")
        model.load_weights("./trained_model/model.h5")
    else:
        print("model.h5 not found")


def main(argv):
    model = AutoEncoder()
    load_model(model)
    model.summary()
    model.compile(loss=MSE, optimizer=Adam(FLAGS.learning_rate))
    train(model)


if __name__ == '__main__':
    if not tf.test.is_gpu_available():
        print("WARNING: Not training with GPU. Training may be slow.")
    app.run(main)
