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


flags.DEFINE_integer("database_size", 500, "number of image to load from the video")
flags.DEFINE_integer("epochs", 20, "number of epochs")
flags.DEFINE_integer("batch_size", 32, "batch size")
flags.DEFINE_integer("nbPixelsPick", 128, "number of pixel pick by batch")  # use 128 in paper
flags.DEFINE_float("learning_rate", 0.001, "learning rate")  # use 0.001 in paper
flags.DEFINE_string("video_train_path", "./video/train.mp4", "video relative path")


def train(model):
    manager = DataManager(frames_skip=200, path=FLAGS.video_train_path, max_img=FLAGS.database_size)
    # manager = DataManager(frames_skip=700, path=FLAGS.video_train_path, max_img=100)
    nbBatches = manager.nbBatches // FLAGS.batch_size
    lossTab = []
    loss = 0

    for epoch in range(1, FLAGS.epochs + 1):
        manager.shuffle()

        for i in tqdm(range(nbBatches), desc="Epoch " + str(epoch) + "/" + str(FLAGS.epochs)):
            (X1, X2), Y = manager.get_batch(FLAGS.batch_size, i)
            X1X2 = np.concatenate((X1, X2), axis=4)

            loss = 0
            samples = np.power(np.random.rand(FLAGS.nbPixelsPick), 2)
            samples = (samples * manager.sizeImages) % manager.sizeImages
            samples = samples.astype(int)

            for b in range(FLAGS.batch_size):
                patchs = []
                pixels = []
                for sample in samples:
                    x = int(manager.motionBatches[b][sample] // manager.width)
                    y = int(manager.motionBatches[b][sample] % manager.width)
                    patchs.append(X1X2[b, x:x + 79, y:y + 79, :, :])
                    pixels.append(Y[b, x + 39:x + 40, y + 39:y + 40, :, :])

                loss += model.train_on_batch(np.array(patchs), np.squeeze(pixels))

            if (i % 5 == 0):
                model.save(FLAGS.model_path)

        lossTab.append(loss / FLAGS.nbPixelsPick)
        print("Epoch {} - loss: {}".format(epoch, lossTab[-1]))

    print("Finished training.")

    model.save(FLAGS.model_path)
    predict(model, video=False, frames_skip=1000)

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
