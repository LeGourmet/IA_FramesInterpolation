import numpy as np
from tqdm import tqdm
import cv2


class DataManager:
    def __init__(self, flag=False):
        # todo un apprentissage judicieux (evoque en session) vous permettra potentiellement de mieux rentabiliser l'apprentissage.
        self.images = None     # save all the images of a video
        self.batches = None    # indices of images X1, Y, X2 for each batches
        self.nbBatches = 0
        self.width = 1280
        self.height = 720
        self.dixImages = flag  # flags true = 8 batches

        imgs = []
        batches = [[], [], []]

        video = cv2.VideoCapture("../video/file_example_MP4_1280_10MG.mp4")
        size = int(video.get(cv2.CAP_PROP_FRAME_COUNT)-1)
        if flag:
            size = min(size, 10)

        for _ in tqdm(range(size)):
            _, frame = video.read()
            imgs.append(cv2.resize(cv2.cvtColor(frame, cv2.IMREAD_COLOR), (self.width, self.height)))
        self.images = np.pad(imgs, ((0,0),(39,39),(39,39),(0,0)), 'constant', constant_values=0)

        self.nbBatches = size-2
        for i in range(self.nbBatches):
            for j in range(3):
                batches[j].append(i+j)
        self.batches = np.array(batches)

    def get_batch(self, batch_size, index=0):
        start = index * batch_size
        end = start + batch_size
        return (self.images[self.batches[0][start:end]], self.images[self.batches[2][start:end]]), self.images[self.batches[1][start:end]]

    def shuffle(self):
        indices = np.random.permutation(self.nbBatches)
        self.batches[0] = self.batches[0][indices]
        self.batches[1] = self.batches[1][indices]
        self.batches[2] = self.batches[2][indices]
