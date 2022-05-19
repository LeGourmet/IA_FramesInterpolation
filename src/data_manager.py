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

        video = cv2.VideoCapture(path)
        size = int(video.get(cv2.CAP_PROP_FRAME_COUNT) - 1)
        size = min(size, min_img) if flag else min(size, max_img)

        video_frames_skip(video, frames_skip)

        for _ in tqdm(range(size), desc="loading images"):
            imgs.append(cv2.resize(read(video), (self.width, self.height)))

        self.images = np.array(imgs)
        self.batches = self.create_batches(size)
        motion = self.estimate_opticalflow()

        self.motionBatches = np.array(motion)
        self.images = np.pad(imgs, ((0, 0), (39, 39), (39, 39), (0, 0)), 'constant', constant_values=0)
        self.images = np.reshape(self.images, (self.images.shape[0], self.images.shape[1], self.images.shape[2], self.images.shape[3], 1))

    def create_batches(self, batches, size):
        batches = [[], [], []]
        self.nbBatches = size - 2
        for i in range(self.nbBatches):
            for j in range(3):
                batches[j].append(i + j)
        return np.array(batches)

    def estimate_opticalflow(self):
        motion = []
        const = 180 / np.pi / 2
        for i in tqdm(range(self.nbBatches), desc="motion calculation"):
            img1 = cv2.cvtColor(self.images[self.batches[0][i]], cv2.COLOR_BGR2GRAY)
            img2 = cv2.cvtColor(self.images[self.batches[2][i]], cv2.COLOR_BGR2GRAY)
            flow = cv2.calcOpticalFlowFarneback(img1, img2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            magnitude, angle = cv2.cartToPolar(flow[:, :, 0], flow[:, :, 1])

            hsv_mask = np.zeros_like(self.images[self.batches[0][i]])
            hsv_mask[:, :, 0] = angle * const
            hsv_mask[:, :, 1] = 255
            hsv_mask[:, :, 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)

            gray_motion = cv2.cvtColor(cv2.cvtColor(hsv_mask, cv2.COLOR_HSV2BGR), cv2.COLOR_BGR2GRAY)
            motion.append(gray_motion.flatten().argsort()[::-1][:])
        return motion

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

    def sample_patches_and_pixels(self, Y, X1X2, samples, b):
        patchs = []
        pixels = []

        for sample in samples:
            x = int(self.motionBatches[b][sample] // self.width)
            y = int(self.motionBatches[b][sample] % self.width)
            patchs.append(X1X2[b, x:x + 79, y:y + 79, :, :])
            pixels.append(Y[b, x + 39:x + 40, y + 39:y + 40, :, :])
        return np.array(patchs), np.squeeze(pixels)

    # samping pixels in the batch according to their motion
    def importance_sampling(self, nbPixelsPick):
        samples = np.power(np.random.rand(nbPixelsPick), FLAGS.importance_sampling)
        samples = (samples * self.sizeImages) % self.sizeImages
        samples = samples.astype(int)
        return samples
