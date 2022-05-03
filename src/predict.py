from utils import setup_cuda_device
setup_cuda_device("0")

import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from src.model import AutoEncoder
from absl import app
from tqdm import tqdm


def predOneImage(model, width, height, iBefore, iAfter):
    size = 600#1500
    X1X2 = np.pad(np.concatenate((iBefore, iAfter), axis=2), ((39,39),(39,39),(0,0)), "constant", constant_values=0)
    img = []
    batch = []

    end = (width-1)*(height-1)
    for x in range(height):
        for y in range(width):
            batch.append(X1X2[x:x+79, y:y+79, :])
            if ((x*width+y)%size == (size-1)) | ((x*y)==end):
                img.extend(model(np.array(batch), training=False))
                batch = []

    return np.reshape(img, (height,width,3))


# video = True => for predict a video ; False only one image
def predict(model, path="../video/video.mkv", video=True):
    # data of video
    video_in = cv2.VideoCapture(path)
    fps = int(video_in.get(cv2.CAP_PROP_FPS))
    nbFrame = int(video_in.get(cv2.CAP_PROP_FRAME_COUNT))-1
    width = int(video_in.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video_in.get(cv2.CAP_PROP_FRAME_HEIGHT))
    nbFrame = 40
    width = 640 #60#400#800#1280
    height = 360 #90#225#450#720

    if video:
        fourcc = cv2.VideoWriter_fourcc(*'HFYU')  # ou *'HFYU' *'XVID' ,*MJPG, *MP4V, *'DIVX' cv2.CV_FOURCC_PROMPT
        video_out = cv2.VideoWriter('../video/output_video.avi', fourcc, fps * 2, (width, height))

        ret, frame1 = video_in.read()
        if ret:
            img1 = cv2.resize(cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB), (width, height))
            img2 = None

            for _ in tqdm(range(nbFrame - 1)):
                _, frame2 = video_in.read()
                img2 = cv2.resize(cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB), (width, height))

                video_out.write(img1)
                video_out.write(predOneImage(model, width, height, img1, img2).astype(np.uint8))
                img1 = img2

            video_out.write(img2, cv2.COLOR_RGB2BGR)
        video_out.release()
        video_in.release()

    else:
        ret1, frame1 = video_in.read()
        ret2, frame2 = video_in.read()
        video_in.release()
        if ret1 & ret2:
            img1 = cv2.resize(cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB), (width, height))
            img2 = cv2.resize(cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB), (width, height))

            _, axes = plt.subplots(1, 3)
            axes[0].imshow(img1 / 255)
            axes[0].set_title("before")
            axes[0].axis('off')

            axes[1].imshow(predOneImage(model, width, height, img2, img2) / 255)
            axes[1].set_title("interp")
            axes[1].axis('off')

            axes[2].imshow(img2 / 255)
            axes[2].set_title("after")
            axes[2].axis('off')

            plt.tight_layout()
            plt.show()


def load_model(model):
    if os.path.isfile("../trained_model/model.h5"):
        print("Loading model from model.h5")
        model.load_weights("../trained_model/model.h5")
    else:
        print("model.h5 not found")
    model.summary()


def main(argv):
    model = AutoEncoder()
    load_model(model)
    predict(model, video=True)


if __name__ == '__main__':
    if not tf.test.is_gpu_available():
        print("WARNING: Not predict with GPU. Predict may be slow.")
    app.run(main)
