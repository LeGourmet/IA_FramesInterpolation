import os
import cv2
import numpy as np
import tensorflow as tf

from src.model import AutoEncoder
from src.data_manager import DataManager

from absl import app

def predict(model):
    cap = cv2.VideoCapture('input_video.mp4')

    # Recupere le nombre d'images par seconde et la dimension    
    fps = video.get(cv2.CAP_PROP_FPS)
    width = vcap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = vcap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    
    # Nous allons generer une video avec 2 fois plus d'images par seconde
    # CV_FOURCC_PROMPT peut etre utilise pour demander a choisir le format
    # de sortie parmi une liste. On choisira un format de sortie *sans perte* (lossless, uncompressed)
    fourcc = CV_FOURCC_PROMPT
    # fourcc = cv2.VideoWriter_fourcc(*'HFYU') # Huffman Lossless Codec
    # fourcc = cv2.VideoWriter_fourcc(*'XVID') # ou *MJPG
    out = cv2.VideoWriter('output_video.avi', fourcc, fps*2, (width, height))    
    
    # On lit la video d'entree tant qu'il y a des images a lire
    hasImages = True
    while hasImages:
        hasImages, img1 = cap.read()
        if hasImages == True:
            # Redimensionner a la resolution voulue
            # img = cv2.resize(img, (frameWidth, frameHeight))
            img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
            hasImages, img2 = cap.read()
            if hasImages == True:
                # Redimensionner a la resolution voulue
                # img = cv2.resize(img, (frameWidth, frameHeight))
                img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
                
                # A ce stade, nous avons deux images, et nous souhaitons generer (predire) une image interpolee I^
                
                imgHat = np.zeros_like(img1) # I^
                
                sizey = img1.shape[0]
                sizex = img1.shape[1]
                
                # On traite tous les pixels
                for y in range(sizey):
                    for x in range(sizex):
                        # Pour chaque coordonnee de pixel, il faut prendre le patch correspondant sur I1 et I2,
                        # (autrement dit, P1 et P2). Appelons ce duo P1-P2 'double patch'.
                        # A un pixel on associe un 'double patch'.
                        #
                        # La premiere dimension dans predict est toujours le batch dans TensorFlow.
                        # Ce qui signifie que vous allez passer PLUSIEURS double-patchs, et cela va produire PLUSIEURS pixels
                        # Cette possibilite est utile pour que le GPU fasse d'un coup les calculs pour de nombreuses donnees
                        # Predire trop de donnees d'un coup peut toutefois saturer la memoire, attention.
                        # Il est possible de parametrer le 'batch_size' quand on appelle 'predict',
                        # pour que TensorFlow decoupe tout seul.
                        # En tout etat de cause, il est indispensable ici d'ecrire votre code de sorte
                        # que plusieurs donnees soient traitees d'un coup.
                
                        ensemble_de_pixels = model.predict(ensemble_de_doublepatchs_P1P2) # ,batch_size=...)
                        ...
                        
                # Passee cette etape, vous avez assemble l'image I^ finale. Il ne reste plus qu'a sauver
                # I1 et I^ dans le flux video de sortie. I2 sera pris en charge a la prochaine iteration,
                # en tant que nouvelle image I1.
                
                out.write(img1)
                out.write(imgHat)
                
    out.release()
    cap.release()


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
    if not tf.test.is_gpu_available():
    app.run(main)
