import cv2
from tqdm import tqdm
import numpy as np


class DataManager:
    def __init__(self, name):
        self.X = [[], [], []]
        self.width = 1280
        self.height = 720
        self.size = None
        self.load_data(name)

    def load_data(self, name):
        imgs = []

        video = cv2.VideoCapture("./video/"+name)
        print("Loading video ...")
        length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        print("Len =", length)
        bar = tqdm(total=100)
        while video.isOpened():
            ret, frame = video.read()
            if not ret:
                break
            imgs.append(cv2.resize(cv2.cvtColor(frame, cv2.IMREAD_COLOR), (self.width, self.height)))
            bar.update(100/length)
        print("Video loaded...")

        self.size = len(imgs)//3
        for i in range(self.size):
            for j in range(3):
                self.X[j].append(imgs[i*self.size + j])
        self.X = np.array(self.X)

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
