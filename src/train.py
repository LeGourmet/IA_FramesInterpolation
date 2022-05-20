import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

from tensorflow.keras.losses import MAE
from tensorflow.keras.optimizers import Adamax
from data_manager import DataManager
from tqdm import tqdm
from absl import app

from utils import *
from model import AutoEncoder

setup_cuda_device("0")

# path
flags.DEFINE_string("video_train_path", "./video/train.mp4", "video relative path")



# size of train data (influence epoch time and train quality)
flags.DEFINE_integer("database_size", 1000, "number of image to load from the video")
flags.DEFINE_integer("nbPixelsPick", 512, "number of pixel pick by batch")

# train parameters
flags.DEFINE_integer("epochs", 2000, "number of epochs")
flags.DEFINE_integer("batch_size", 64, "batch size")
flags.DEFINE_float("learning_rate", 0.001, "learning rate")

# impoortance of pixel motion in trainingD
flags.DEFINE_float("sampling_factor", 3.0, "sample moving pixels more often")

def train(model):
    manager = DataManager(frames_skip=100, path=FLAGS.video_train_path, max_img=FLAGS.database_size)
    nbBatchs = manager.nbGroups // FLAGS.batch_size
    lossTab = []
    loss = 0

    for epoch in range(1, FLAGS.epochs + 1):
        manager.shuffle()  # shuffle the whole database at the start of each epoch

        description = "Epoch " + str(epoch) + "/" + str(FLAGS.epochs)
        for i in tqdm(range(nbBatchs), desc=description):
            loss = 0  # loss for current batch

            (X1, X2), Y = manager.get_batch(FLAGS.batch_size, i)
            X1X2 = np.concatenate((X1, X2), axis=4)
            samples = manager.importance_sampling(FLAGS.nbPixelsPick, FLAGS.sampling_factor)

            for batch in range(FLAGS.batch_size):
                patchs, pixels = manager.sample_patches_and_pixels(Y, X1X2, samples, batch)
                loss += model.train_on_batch(patchs, pixels)

            if (i % 5 == 0):
                model.save(FLAGS.model_path)

        lossTab.append(loss / FLAGS.batch_size)  # average loss per epoch
        print("Epoch {} - loss: {}".format(epoch, lossTab[-1]))

    print("Finished training.")
    model.save(FLAGS.model_path)
    show_plot(lossTab)


def show_plot(lossTab):
    plt.plot(lossTab)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()


def main(argv):
    model = AutoEncoder()
    load_model(model)
    model.compile(loss=MAE, optimizer=Adamax(FLAGS.learning_rate))
    train(model)


if __name__ == '__main__':
    if not tf.test.is_gpu_available():
        print("WARNING: Not training with GPU. Training may be slow.")
    app.run(main)
