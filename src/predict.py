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
        print(img1.shape)

        for _ in range(fps-1):
            _, frame2 = video_in.read()
            img2 = cv2.resize(cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB), (width, height))
            X1X2 = np.concatenate((img1, img2), axis=2)
            X1X2 = np.reshape(X1X2,(1,X1X2.shape[0],X1X2.shape[1],X1X2.shape[2]))
            bar = tqdm(total=width*height)
            for x in range(39,height+38):
                for y in range(39,width+38):
                    pix = model.predict(X1X2[:,x-39:x+40, y-39:y+40, :])
                    bar.update()
            video_out.write(img1)
            video_out.write(imgPred)
            img1 = img2

    video_out.release()
    video_in.release()


def predictImage(model):
    video_in = cv2.VideoCapture("../video/video.mkv")
    width = 1280
    height = 720

    ret1, frame1 = video_in.read()
    ret2, frame2 = video_in.read()
    video_in.release()
    if ret1 & ret2:
        img1 = cv2.resize(cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB), (width, height))
        imgPred = []
        img2 = cv2.resize(cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB), (width, height))
        X1X2 = np.concatenate((img1, img2), axis=2)
        X1X2 = np.pad(X1X2,((39,39),(39,39),(0,0)))

        bar = tqdm(total=width*height)
        for x in range(39, height+39):
            batch = []
            for y in range(39, width+39):
                batch.append(X1X2[x - 39:x + 40, y - 39:y + 40, :])
                bar.update()
            batch = np.array(batch)
            imgPred.append([model.predict(batch)])

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
