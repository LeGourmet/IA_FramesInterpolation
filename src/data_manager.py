from utils import *
import numpy as np
from tqdm import tqdm
import cv2

class DataManager:
    def __init__(self, flag=False, min_img=10, max_img=1000, frames_skip=0, path=""):
        self.images = None     # save all the images of a video
        self.batches = None    # indices of images X1, Y, X2 for each batches
        self.nbBatches = 0
        self.width = 1280
        self.height = 720

        imgs = []
        batches = [[], [], []]

        video = cv2.VideoCapture(path)
        size = int(video.get(cv2.CAP_PROP_FRAME_COUNT)-1)
        size = min(size, min_img) if flag else min(size, max_img)

        video_frames_skip(video, frames_skip)

        for _ in tqdm(range(size)):
            imgs.append(cv2.resize(read(video), (self.width, self.height)))

        self.images = np.pad(imgs, ((0, 0), (39, 39), (39, 39), (0, 0)), 'constant', constant_values=0)
        self.images = np.reshape(self.images, (self.images.shape[0], self.images.shape[1], self.images.shape[2], self.images.shape[3], 1))

        self.nbBatches = size - 2
        for i in range(self.nbBatches):
            for j in range(3):
                batches[j].append(i + j)
        self.batches = np.array(batches)

    def get_batch(self, batch_size, index=0):
        # add batch in the other sens (time)
        # swap horizontal / in the other sense
        # swap vertical / in the other sense
        # swap the two / in the other sense
        start = index * batch_size
        end = start + batch_size
        return (self.images[self.batches[0][start:end]], self.images[self.batches[2][start:end]]), self.images[self.batches[1][start:end]]

    def shuffle(self):
        indices = np.random.permutation(self.nbBatches)
        self.batches[0] = self.batches[0][indices]
        self.batches[1] = self.batches[1][indices]
        self.batches[2] = self.batches[2][indices]
