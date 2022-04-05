import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import tensorflow as tf

physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

from absl import app
from absl import flags
from tqdm import tqdm
import matplotlib.pyplot as plt
from data_manager import DataManager
from model import buildModel, myLossFnct
from sample import sample
from tensorflow.keras.optimizers import Adam

flags.DEFINE_integer("epochs", 100, "number of epochs")
flags.DEFINE_integer("batch_size", 32, "batch size")
flags.DEFINE_float("learning_rate", 0.005, "learning rate")
FLAGS = flags.FLAGS


def train(model):
    manager = DataManager("BlacKkKlansman.mkv")
    nbBatches = manager.size // FLAGS.batch_size
    lossTab = []
    loss = 0

    for epoch in range(1,FLAGS.epochs+1):
        print('Epoch', epoch, '/', FLAGS.epochs)
        manager.shuffle()

        for i in tqdm(range(nbBatches)):
            X1, Y, X2 = manager.get_batch(FLAGS.batch_size, i)

            for x in range(manager.width):
                for y in range(manager.height):
                    R = subImage(X1,79,x,y)+subImage(X2,79,x,y)
                    P = subImage(X1,41,x,y)+subImage(X2,41,x,y)
                    loss = model.train_on_batch(R, (P, subImage(Y,1,x,y)))

        lossTab.append(loss)
        print("Epoch {} - loss: {}".format(epoch, loss))

    print("Finished training.")

    # model.save("./trained_model/model.h5")
    sample(model)

    plt.plot(lossTab)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()


def subImage(batch, size, x, y):
    sX = x-(size//2)
    sY = y-(size//2)
    a = batch[sX:(sX+size),sY:(sY+size)]
    print(a.shape)
    return a


def main(argv):
    model = buildModel()

    if os.path.isfile("./trained_model/model.h5"):
        print("Loading model from model.h5")
        model.load_weights("./trained_model/model.h5")
    else:
        print("model.h5 not found")

    model.compile(loss=myLossFnct, optimizer=Adam(FLAGS.learning_rate))
    model.summary()

    train(model)


if __name__ == '__main__':
    if not tf.test.is_gpu_available():
        print("WARNING: Not training with GPU. Training may be slow.")
    app.run(main)
