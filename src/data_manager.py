import numpy as np
from tqdm import tqdm
import cv2


class DataManager:
    def __init__(self, flag=False):
        # todo un apprentissage judicieux (evoque en session) vous permettra potentiellement de mieux rentabiliser l'apprentissage.
        self.X = None  # (3 frames, batches, height, width, color)
        self.width = 1280
        self.height = 720
        self.size = None

        # todo Si ce drapeau est defini, votre code ne devra prendre en compte que les 10 premieres images de la video
        # pour l'apprentissage. Cette fonctionnalite est obligatoire (voir sujet du projet).
        self.dixImages = flag

        imgs = []
        data = [[], [], []]

        video = cv2.VideoCapture("./src/video/video.mkv")
        length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        bar = tqdm(total=length)
        while video.isOpened():
            ret, frame = video.read()
            if not ret:
                break
            imgs.append(cv2.resize(cv2.cvtColor(frame, cv2.IMREAD_COLOR), (self.width, self.height)))
            bar.update(1)

        self.size = len(imgs) // 3
        for i in range(self.size):
            for j in range(3):
                data[j].append(imgs[i * 3 + j])
        self.X = np.array(data)

    def get_batch(self, batch_size, index=0):
        start = index * batch_size
        end = start + batch_size
        return (self.X[0][start:end], self.X[2][start:end]), self.X[1][start:end]

    def shuffle(self):
        indices = np.arange(self.size)
        np.random.shuffle(indices)
        self.X[0] = self.X[0][indices]
        self.X[1] = self.X[1][indices]
        self.X[2] = self.X[2][indices]
