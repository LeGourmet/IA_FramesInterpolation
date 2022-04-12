import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import tensorflow as tf

physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

import numpy as np
from absl import app
from tqdm import tqdm
from absl import flags
import matplotlib.pyplot as plt
#from src.predict import predict
from src.model import AutoEncoder
from src.data_manager import DataManager
from tensorflow.keras.optimizers import Adam

flags.DEFINE_integer("epochs", 100, "number of epochs")
flags.DEFINE_integer("batch_size", 32, "batch size")
flags.DEFINE_float("learning_rate", 0.005, "learning rate")
FLAGS = flags.FLAGS


def train(model, optimizer):
    manager = DataManager()
    mse = tf.keras.losses.MeanSquaredError()
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
                    P = np.concatenate((subImage(X1,41,x,y),subImage(X2,41,x,y)), axis=2)  # shape = (None,41,82,3)
                    pix = subImage(Y,1,x,y)                                                # shape = (None,1,1,3)

                    with tf.GradientTape() as g:
                        pix_pred = applyKernel(model(R), P)
                        loss = mse(pix,pix_pred)
                    grad = g.gradient(loss, model.trainable_variables)
                    optimizer.apply_gradients(zip(grad, model.trainable_variables))

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
    model.summary()
    train(model, Adam(FLAGS.learning_rate))


if __name__ == '__main__':
    if not tf.test.is_gpu_available():
        print("WARNING: Not training with GPU. Training may be slow.")
    app.run(main)
