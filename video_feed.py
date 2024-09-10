import cv2
from PIL import Image, ImageTk
import threading
from database import Database
import os

class VideoFeed:
    def __init__(self, label_frame):
        self.label_frame = label_frame
        self.video_running = False
        self.face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
        self.clf = cv2.face.LBPHFaceRecognizer_create()
        self.load_classifier()
        self.db = Database()

    def load_classifier(self):
        if os.path.exists("classifier.xml"):
            self.clf.read("classifier.xml")
        else:
            self.clf = None

    def toggle_video(self):
        if self.video_running:
            self.video_running = False
        else:
            self.video_running = True
            self.start_video()

    def draw_boundary(self, img, classifier, scaleFactor, minNeighbors):
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        features = classifier.detectMultiScale(gray_image, scaleFactor, minNeighbors)
        coords = []

        for (x, y, w, h) in features:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), 2)
            id, pred = self.clf.predict(gray_image[y:y + h, x:x + w])
            confidence = int(100 * (1 - pred / 300))

            name = self.db.get_name_by_id(id)
            if name and confidence > 70:
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(img, name, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 1, cv2.LINE_AA)
            else:
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(img, 'unknown', (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 1, cv2.LINE_AA)

            coords = [x, y, w, h]
        return coords

    def recognize(self, img):
        coords = self.draw_boundary(img, self.face_cascade, 1.1, 10)
        return img

    def start_video(self):
        video_capture = cv2.VideoCapture(0)
        video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
        video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)

        def update_frame():
            if self.video_running:
                ret, frame = video_capture.read()
                if ret:
                    frame = cv2.flip(frame, 1)
                    if self.clf:
                        frame = self.recognize(frame)
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    img = Image.fromarray(frame)
                    imgtk = ImageTk.PhotoImage(image=img)
                    self.label_frame.imgtk = imgtk
                    self.label_frame.configure(image=imgtk)
                self.label_frame.after(10, update_frame)
            else:
                video_capture.release()
                self.label_frame.configure(image='')

        update_frame()
