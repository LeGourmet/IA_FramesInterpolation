import cv2
import numpy as np

from tqdm import tqdm

from utils import *


class DataManager:
    def __init__(self, flag=False, min_img=10, max_img=1000, frames_skip=0, path=""):
        self.images = None         # save all the images of a video
        self.groups = None         # indices of images X1, Y, X2
        self.motionGroups = None  # motion of all pixels for each batches
        self.nbGroups = 0
        self.width = 1280
        self.height = 720
        self.sizeImages = self.width * self.height

        video = cv2.VideoCapture(path)
        size = int(video.get(cv2.CAP_PROP_FRAME_COUNT) - 1)
        size = min(size, min_img) if flag else min(size, max_img)
        video_frames_skip(video, frames_skip)

        imgs = []
        for _ in tqdm(range(size), desc="loading images"):
            imgs.append(cv2.resize(read(video), (self.width, self.height)))

        self.images = np.array(imgs)
        self.groups = self.create_groups(size)

        self.motionGroups = self.estimate_opticalflow()
        self.images = np.pad(imgs, ((0, 0), (39, 39), (39, 39), (0, 0)), 'constant', constant_values=0)
        self.images = np.reshape(self.images, (self.images.shape[0], self.images.shape[1], self.images.shape[2], self.images.shape[3], 1))

    def create_groups(self, size):
        """Create groups of images (X1, Y, X2)"""
        groups = [[], [], []]
        self.nbGroups = size - 2
        for i in range(self.nbGroups):
            for j in range(3):
                groups[j].append(i + j)
        return np.array(groups)

    def estimate_opticalflow(self):
        """use cv2 optical flow to estimate the motion of each pixel
        then use the magnitude of motion to sort pixel according to their motion in a batch
        the pixels will latter be sampled according to their motion
        based on the intuitive article : https://www.geeksforgeeks.org/python-opencv-dense-optical-flow/

        Returns:
            motion: permutation list pixel in a batch ordered by moving importance
        """
        motion = []
        const = 180 / np.pi / 2
        for i in tqdm(range(self.nbGroups), desc="motion calculation"):
            img1 = cv2.cvtColor(self.images[self.groups[0][i]], cv2.COLOR_BGR2GRAY)
            img2 = cv2.cvtColor(self.images[self.groups[2][i]], cv2.COLOR_BGR2GRAY)
            flow = cv2.calcOpticalFlowFarneback(img1, img2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            magnitude, angle = cv2.cartToPolar(flow[:, :, 0], flow[:, :, 1])

            hsv_mask = np.zeros_like(self.images[self.groups[0][i]])
            hsv_mask[:, :, 0] = angle * const
            hsv_mask[:, :, 1] = 255
            hsv_mask[:, :, 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)

            gray_motion = cv2.cvtColor(cv2.cvtColor(hsv_mask, cv2.COLOR_HSV2BGR), cv2.COLOR_BGR2GRAY)
            motion.append(gray_motion.flatten().argsort()[::-1][:])
        return np.array(motion)

    def get_batch(self, batch_size, index=0):
        """retrun batch_size number of groups starting from index"""
        start = index * batch_size
        end = start + batch_size
        return (self.images[self.groups[0][start:end]], self.images[self.groups[2][start:end]]), self.images[self.groups[1][start:end]]

    def shuffle(self):
        indices = np.random.permutation(self.nbGroups)
        self.groups[0] = self.groups[0][indices]
        self.groups[1] = self.groups[1][indices]
        self.groups[2] = self.groups[2][indices]
        self.motionGroups = self.motionGroups[indices]

    def sample_patches_and_pixels(self, Y, X1X2, samples, batch):
        """sample pixels and patches in the batch according to their motion

        Args:
            Y : true image
            X1X2 : concatenated images X1 & X2
            samples [] : an array of indices of pixels to sample
            batch (int): index of individual batch inside batch_size
        Returns:
            (tuple) : (training patches , true pixels)
        """
        patchs = []
        pixels = []

        for sample in samples:
            x = int(self.motionGroups[batch][sample] // self.width)
            y = int(self.motionGroups[batch][sample] % self.width)
            patchs.append(X1X2[batch, x:x + 79, y:y + 79, :, :])
            pixels.append(Y[batch, x + 39:x + 40, y + 39:y + 40, :, :])
        return np.array(patchs), np.squeeze(pixels)

    # samping pixels in the batch according to their motion
    def importance_sampling(self, nbPixelsPick, factor):
        """sample pixels in the batch according to their motion

        Args:
            nbPixelsPick (int): number of pixels to select
            factor (float): how much to weight the importance of the motion
        Returns:
            array : indices of pixels to sample
        """
        samples = np.power(np.random.rand(nbPixelsPick), factor)
        samples = (samples * self.sizeImages) % self.sizeImages
        samples = samples.astype(int)
        return samples
