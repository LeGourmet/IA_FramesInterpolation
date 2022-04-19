import os
import cv2
import numpy as np
from tqdm import tqdm

from src.model import AutoEncoder
from src.data_manager import DataManager

from absl import app


def predict(model):
    video_in = cv2.VideoCapture("../video/video.mkv")
    # data of input video
    fps = int(video_in.get(cv2.CAP_PROP_FPS))
    width = int(video_in.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video_in.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # create output video
    fourcc = cv2.VideoWriter_fourcc(*'HFYU') # ou *'XVID' ,*MJPG, *MP4V, cv2.CV_FOURCC_PROMPT
    video_out = cv2.VideoWriter('../video/output_video.avi', fourcc, fps*2, (width, height))

    ret, frame1 = video_in.read()
    if ret:
        img1 = cv2.resize(cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB), (width, height))
        imgPred = np.zeros_like(img1)

        for _ in tqdm(range(fps-1)):
            _, frame2 = video_in.read()
            img2 = cv2.resize(cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB), (width, height))
            X1X2 = np.concatenate((img1, img2), axis=2)
            for y in range(width):
                for x in range(height):
                    pix = model.predict(X1X2[x-39:x+40, y-39:y+40, :])
                    print(pix)
            video_out.write(img1)
            video_out.write(imgPred)
            img1 = img2

    video_out.release()
    video_in.release()


def load_model(model):
    if os.path.isfile("./trained_model/model.h5"):
        print("Loading model from model.h5")
        model.load_weights("./trained_model/model.h5")
    else:
        print("model.h5 not found")
    model.summary()


def main(argv):
    model = AutoEncoder()
    load_model(model)
    predict(model)


if __name__ == '__main__':
    app.run(main)
