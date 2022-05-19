import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

from tensorflow.keras.losses import MAE
from tensorflow.keras.optimizers import Adamax
from data_manager import DataManager
from random import randint
from tqdm import tqdm
from absl import app

from utils import *
from model import AutoEncoder
from predict import predict

setup_cuda_device("0")

flags.DEFINE_string("video_train_path", "./video/train.mp4", "video relative path")

flags.DEFINE_integer("epochs", 200, "number of epochs")
flags.DEFINE_integer("database_size", 100, "number of image to load from the video")
flags.DEFINE_float("importance_sampling", 2.0, "sample moving pixels more often")  # use 0.001 in paper

flags.DEFINE_integer("batch_size", 32, "batch size")
flags.DEFINE_integer("nbPixelsPick", 128, "number of pixel pick by batch")  # use 128 in paper
flags.DEFINE_float("learning_rate", 0.001, "learning rate")  # use 0.001 in paper


def train(model):
    manager = DataManager(frames_skip=100, path=FLAGS.video_train_path, max_img=FLAGS.database_size)
    # manager = DataManager(frames_skip=700, path=FLAGS.video_train_path, max_img=100)
    nbBatches = manager.nbBatches // FLAGS.batch_size
    lossTab = []
    loss = 0

    for epoch in range(1, FLAGS.epochs + 1):
        manager.shuffle()

        for i in tqdm(range(nbBatches), desc="Epoch " + str(epoch) + "/" + str(FLAGS.epochs)):
            loss = 0

            (X1, X2), Y = manager.get_batch(FLAGS.batch_size, i)
            X1X2 = np.concatenate((X1, X2), axis=4)

            # samping pixels in the batch according to their motion
            samples = np.power(np.random.rand(FLAGS.nbPixelsPick), FLAGS.importance_sampling)
            samples = (samples * manager.sizeImages) % manager.sizeImages
            samples = samples.astype(int)

            #
            for b in range(FLAGS.batch_size):
                patchs, pixels = manager.sample_patches_and_pixels(Y, X1X2, samples, b)

                loss += model.train_on_batch(patchs, pixels)

            if (i % 5 == 0):
                model.save(FLAGS.model_path)

        lossTab.append(loss / FLAGS.batch_size)
        print("Epoch {} - loss: {}".format(epoch, lossTab[-1]))

    print("Finished training.")

    model.save(FLAGS.model_path)
    predict(model, video=False, frames_skip=100)

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
