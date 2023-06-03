import os
import tensorflow as tf

from tqdm import tqdm
from absl import flags
from absl.flags import FLAGS

flags.DEFINE_string("model_path", "./trained_model/model.h5", "model relative path")


def setup_cuda_device(gpus: str = "-1"):
    """ force enable or disable cuda devices"""
    os.environ['CUDA_VISIBLE_DEVICES'] = gpus
    physical_devices = tf.config.list_physical_devices('GPU')
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)
    print("Num GPUs Available: ", len(physical_devices))


def video_frames_skip(video, nb):
    for _ in tqdm(range(nb), desc="frame skiping"):
        video.grab()


def read(video):
    _, frame = video.read()
    return frame


def load_model(model):
    if os.path.isfile(FLAGS.model_path):
        print("Loading model from model.h5")
        model.load_weights(FLAGS.model_path)
    else:
        print("model.h5 not found")
    model.summary()
