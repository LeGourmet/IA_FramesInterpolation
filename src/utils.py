import os
import tensorflow as tf


def setup_cuda_device(gpus: str = "-1"):
    """ force enable or disable cuda devices"""
    os.environ['CUDA_VISIBLE_DEVICES'] = gpus
    physical_devices = tf.config.list_physical_devices('GPU')
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)
    print("Num GPUs Available: ", len(physical_devices))
