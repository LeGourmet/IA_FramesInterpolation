from utils import setup_cuda_device
setup_cuda_device("0")

import os
import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from src.model import AutoEncoder

from absl import app


def predictVideo(model):
    video_in = cv2.VideoCapture("../video/video.mkv")
    # data of input video
    nbFrame = int(video_in.get(cv2.CAP_PROP_FRAME_COUNT)-1)  # todo investigate -1 ? dm same
    nbFrame = 10
    width = int(video_in.get(cv2.CAP_PROP_FRAME_WIDTH))
    width = 160#400#800#1280
    height = int(video_in.get(cv2.CAP_PROP_FRAME_HEIGHT))
    height = 90#225#450#720

    # create output video
    fourcc = cv2.VideoWriter_fourcc(*'HFYU')  # ou *'HFYU' *'XVID' ,*MJPG, *MP4V, *'DIVX' cv2.CV_FOURCC_PROMPT
    video_out = cv2.VideoWriter('../video/output_video.avi', fourcc, int(video_in.get(cv2.CAP_PROP_FPS))*2, (width, height))

    ret, frame1 = video_in.read()
    if ret:
        img1 = cv2.resize(cv2.cvtColor(frame1, cv2.IMREAD_COLOR), (width, height))
        img2 = None

        for _ in tqdm(range(nbFrame-1)):
            _, frame2 = video_in.read()
            img2 = cv2.resize(cv2.cvtColor(frame2, cv2.IMREAD_COLOR), (width, height))
            X1X2 = np.concatenate((img1, img2), axis=2)
            X1X2 = np.pad(X1X2,((39,39),(39,39),(0,0)), "constant", constant_values=0)
            imgPred = []

            for x in range(39, height + 39): # optimize with begest
                batch = []
                for y in range(39, width + 39):
                    batch.append(X1X2[x - 39:x + 40, y - 39:y + 40, :])
                imgPred.append(model.predict(np.array(batch)))

            #video_out.write(img1)
            video_out.write(np.array(imgPred).astype(np.uint8))
            img1 = img2

        video_out.write(img2)  # todo marche pas
    video_out.release()
    video_in.release()


def predictImage(model):
    video_in = cv2.VideoCapture("../video/file_example_MP4_1280_10MG.mp4")
    width = 128
    height = 72

    ret1, frame1 = video_in.read()
    ret2, frame2 = video_in.read()
    video_in.release()
    if ret1 & ret2:
        img1 = cv2.resize(cv2.cvtColor(frame1, cv2.IMREAD_COLOR), (width, height))
        img2 = cv2.resize(cv2.cvtColor(frame2, cv2.IMREAD_COLOR), (width, height))
        imgPred = []
        X1X2 = np.concatenate((img1, img2), axis=2)
        X1X2 = np.pad(X1X2,((39,39+6),(39,39+10),(0,0)), "constant", constant_values=0)

        bar = tqdm(total=width*height)
        for x in range(39+6, height+39+6):
            batch = []
            for y in range(39+10, width+39+10):
                batch.append(X1X2[(x-39):(x+40), y - 39:y + 40, :])
                bar.update()
            imgPred.append(model.predict(np.array(batch)))

        _, axes = plt.subplots(3, 1)
        axes[0].imshow(img1/255)
        axes[0].set_title("before")
        axes[0].axis('off')

        axes[1].imshow(np.squeeze(np.array(imgPred))/255)
        axes[1].set_title("interp")
        axes[1].axis('off')

        axes[2].imshow(img2/255)
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
    predictImage(model)
    video_in = cv2.VideoCapture("../video/output_video.avi")
    nbFrame = int(video_in.get(cv2.CAP_PROP_FRAME_COUNT) - 1)
    print(nbFrame)


if __name__ == '__main__':
    app.run(main)
