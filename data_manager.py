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
        print("Nombre frame =", length)
        bar = tqdm(total=100)
        while video.isOpened():
            ret, frame = video.read()
            if not ret:
                break
            imgs.append(cv2.resize(cv2.cvtColor(frame, cv2.IMREAD_COLOR), (self.width, self.height)))
            bar.update(100/length)

        self.size = len(imgs)//3
        for i in range(self.size):
            for j in range(3):
                data[j].append(imgs[i*3 + j])
        self.X = np.array(data)
        print(self.X.shape)

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
