import cv2
import numpy as np

from tqdm import tqdm

from utils import *


class DataManager:
    def __init__(self, flag=False, min_img=10, max_img=1000, frames_skip=0, path=""):
        self.images = None         # save all the images of a video
        self.batches = None        # indices of images X1, Y, X2 for each batches
        self.motionBatches = None  # motion of all pixels for each batches
        self.nbBatches = 0
        self.width = 1280
        self.height = 720
        self.sizeImages = self.width * self.height

        imgs = []
        motion = []
        batches = [[], [], []]

        video = cv2.VideoCapture(path)
        size = int(video.get(cv2.CAP_PROP_FRAME_COUNT) - 1)
        size = min(size, min_img) if flag else min(size, max_img)

        video_frames_skip(video, frames_skip)

        for _ in tqdm(range(size), desc="loading images"):
            imgs.append(cv2.resize(read(video), (self.width, self.height)))

        self.images = np.array(imgs)

        self.nbBatches = size - 2
        for i in range(self.nbBatches):
            for j in range(3):
                batches[j].append(i + j)
        self.batches = np.array(batches)

        for i in tqdm(range(self.nbBatches), desc="motion calculation"):
            img1 = cv2.cvtColor(self.images[self.batches[0][i]], cv2.COLOR_BGR2GRAY)
            img2 = cv2.cvtColor(self.images[self.batches[2][i]], cv2.COLOR_BGR2GRAY)
            flow = cv2.calcOpticalFlowFarneback(img1, img2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            magnitude, _ = cv2.cartToPolar(flow[:, :, 0], flow[:, :, 1])
            motion.append(magnitude.flatten().argsort()[:])

        self.motionBatches = np.array(motion)
        self.images = np.pad(imgs, ((0, 0), (39, 39), (39, 39), (0, 0)), 'constant', constant_values=0)
        self.images = np.reshape(self.images, (self.images.shape[0], self.images.shape[1], self.images.shape[2], self.images.shape[3], 1))

    def get_batch(self, batch_size, index=0):
        start = index * batch_size
        end = start + batch_size
        return (self.images[self.batches[0][start:end]], self.images[self.batches[2][start:end]]), self.images[self.batches[1][start:end]]

    def shuffle(self):
        indices = np.random.permutation(self.nbBatches)
        self.batches[0] = self.batches[0][indices]
        self.batches[1] = self.batches[1][indices]
        self.batches[2] = self.batches[2][indices]
        self.motionBatches = self.motionBatches[indices]

    def get_patchs_and_pixels(self, X1X2, Y, batch_size, r):
        patchs = []
        pixels = []

        for b in range(batch_size):
            x = int(self.motionBatches[b][r] // self.width)
            y = int(self.motionBatches[b][r] % self.width)
            patchs.append(X1X2[b, x:x + 79, y:y + 79, :, :])
            pixels.append(Y[b, x + 39:x + 40, y + 39:y + 40, :, :])

        return np.array(patchs), np.squeeze(pixels)
