import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from absl import app
from tqdm import tqdm

from model import AutoEncoder
from utils import *

setup_cuda_device("0")

flags.DEFINE_bool("video", True, "predict a video or just an image")
flags.DEFINE_integer("frames_skip", 0, "nb frame to skip")
flags.DEFINE_integer("frame_to_load_in_memory", 3000, "nb of frame to load in VRAM")
flags.DEFINE_string("video_output_path", './video/output_video.avi', "path of the output video")
flags.DEFINE_string("video_predict_path", "./video/bbb_720p_7440-13850.mp4", "video relative path")


def predOneImage(model, width, height, iBefore, iAfter):
    size = FLAGS.frame_to_load_in_memory
    iBefore = np.reshape(iBefore, (iBefore.shape[0], iBefore.shape[1], iBefore.shape[2], 1))
    iAfter = np.reshape(iAfter, (iAfter.shape[0], iAfter.shape[1], iAfter.shape[2], 1))
    X1X2 = np.pad(np.concatenate((iBefore, iAfter), axis=3), ((39,39),(39,39),(0,0),(0,0)), "constant", constant_values=0)
    img = []
    batch = []

    bar = tqdm(total=height*width)
    for x in range(height):
        for y in range(width):
            batch.append(X1X2[x:x+79, y:y+79, :, :])
            if len(batch) == size:
                img.extend(model.predict_on_batch(np.array(batch)))
                bar.update(len(batch))
                batch = []
    img.extend(model.predict_on_batch(np.array(batch)))
    bar.update(len(batch))

    return np.reshape(img, (height,width,3))


# video = True => for predict a video ; False only one image
def predict(model, video=False, frames_skip=0):
    video_in = cv2.VideoCapture(FLAGS.video_predict_path)
    fps = int(video_in.get(cv2.CAP_PROP_FPS))
    nbFrame = int(video_in.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
    # width = int(video_in.get(cv2.CAP_PROP_FRAME_WIDTH))
    # height = int(video_in.get(cv2.CAP_PROP_FRAME_HEIGHT))
    nbFrame = 20
    res_mult = 1/8
    width = int(1280 * res_mult)
    height = int(720 * res_mult)

    video_frames_skip(video_in, frames_skip)

    if video:
        fourcc = cv2.VideoWriter_fourcc(*'FFV1')
        video_out = cv2.VideoWriter(FLAGS.video_output_path, fourcc, fps * 2, (width, height))

        ret, frame1 = video_in.read()
        if ret:
            img1 = cv2.resize(frame1, (width, height))
            img2 = None

            for i in range(1, nbFrame):
                print("frame", i, "/", nbFrame - 1)
                _, frame2 = video_in.read()
                img2 = cv2.resize(frame2, (width, height))

                video_out.write(img1)
                video_out.write(predOneImage(model, width, height, img1, img2).astype(np.uint8))
                img1 = img2

            video_out.write(img2)
        video_out.release()
        video_in.release()

    else:
        ret1, frame1 = video_in.read()
        ret2, frame2 = video_in.read()
        video_in.release()
        if ret1 & ret2:
            img1 = cv2.resize(frame1, (width, height))
            img2 = cv2.resize(frame2, (width, height))

            _, axes = plt.subplots(1, 3)
            axes[0].imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB) / 255)
            axes[0].set_title("before")
            axes[0].axis('off')

            axes[1].imshow(cv2.cvtColor(predOneImage(model, width, height, img1, img2), cv2.COLOR_BGR2RGB) / 255)
            axes[1].set_title("interp")
            axes[1].axis('off')

            axes[2].imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB) / 255)
            axes[2].set_title("after")
            axes[2].axis('off')

            plt.tight_layout()
            plt.show()


def main(argv):
    model = AutoEncoder()
    load_model(model)
    predict(model, video=FLAGS.video, frames_skip=FLAGS.frames_skip)


if __name__ == '__main__':
    if not tf.test.is_gpu_available():
        print("WARNING: Not predict with GPU. Predict may be slow.")
    app.run(main)
