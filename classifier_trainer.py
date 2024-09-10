import os
import cv2
import numpy as np
from PIL import Image
from tkinter import messagebox

class ClassifierTrainer:
    def train(self):
        data_dir = "D:/Face_detection/data"
        path = [os.path.join(data_dir, f) for f in os.listdir(data_dir)]
        faces = []
        ids = []

        for image in path:
            img = Image.open(image).convert('L')
            imageNp = np.array(img, 'uint8')
            id = os.path.split(image)[1].split(".")[1]
            faces.append(imageNp)
            ids.append(int(id))  # Convert id to int for compatibility

        ids = np.array(ids, dtype=np.int32)

        clf = cv2.face.LBPHFaceRecognizer_create()
        clf.train(faces, ids)
        clf.write("classifier.xml")
        messagebox.showinfo('Result', 'Training dataset completed!!!')
