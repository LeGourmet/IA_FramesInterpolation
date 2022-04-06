import cv2
from tqdm import tqdm
import numpy as np


class DataManager:
    def __init__(self, name):
        self.X = None  # (3 frames, 484 batches, height, width, color)
        self.width = 1280
        self.height = 720
        self.size = None
        self.load_data(name)

    def load_data(self, name):
        imgs = []
        data = [[], [], []]

        video = cv2.VideoCapture("./video/"+name)
        length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        bar = tqdm(total=length)
        while video.isOpened():
            ret, frame = video.read()
            if not ret:
                break
            imgs.append(cv2.resize(cv2.cvtColor(frame, cv2.IMREAD_COLOR), (self.width, self.height)))
            bar.update(1)

        self.size = len(imgs)//3
        for i in range(self.size):
            for j in range(3):
                data[j].append(imgs[i*3 + j])
        self.X = np.array(data)

    def get_batch(self, batch_size, index=0):
        start = index * batch_size
        end = start + batch_size
        return self.X[0][start:end], self.X[1][start:end], self.X[2][start:end]

    def shuffle(self):
        indices = np.arange(self.size)
        np.random.shuffle(indices)
        self.X[0] = self.X[0][indices]
        self.X[1] = self.X[1][indices]
        self.X[2] = self.X[2][indices]


if __name__ == '__main__':
    dm = DataManager("Projet_WebGL_2019.mp4")
    print("size =", dm.size)
    print("data shape =", dm.X.shape)
    X1, Y, X2 = dm.get_batch(32)
    print("X1 shape =", X1.shape)
    print("Y shape =", Y.shape)
    print("X2 shape =", X2.shape)
    dm.shuffle()
    print("data shape =", dm.X.shape)
