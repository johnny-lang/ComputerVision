import cv2
import threading
from PIL import Image, ImageTk
from tkinter import messagebox
from database import Database

class DatasetGenerator:
    def __init__(self, mssv_entry, name_entry, age_entry, label_frame):
        self.mssv_entry = mssv_entry
        self.name_entry = name_entry
        self.age_entry = age_entry
        self.label_frame = label_frame
        self.db = Database()

    def face_cropped(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = cv2.CascadeClassifier("haarcascade_frontalface_default.xml").detectMultiScale(gray, 1.3, 5)
        if len(faces) == 0:
            return None
        for (x, y, w, h) in faces:
            cropped_face = img[y:y + h, x:x + w]
        return cropped_face

    def capture_faces(self):
        cap = cv2.VideoCapture(0)
        img_id = 0
        user_id = self.mssv_entry.get()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            cropped_face = self.face_cropped(frame)
            if cropped_face is not None:
                img_id += 1
                face = cv2.resize(cropped_face, (120, 200))
                face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                file_name_path = f"D:/Face_detection/data/user.{user_id}.{img_id}.jpg"
                cv2.imwrite(file_name_path, face)
                cv2.putText(face, str(img_id), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

                face_rgb = cv2.cvtColor(face, cv2.COLOR_GRAY2RGB)
                img = Image.fromarray(face_rgb)
                img = img.resize((300, 400), Image.LANCZOS)
                imgtk = ImageTk.PhotoImage(image=img)
                self.label_frame.imgtk = imgtk
                self.label_frame.configure(image=imgtk)

            if cv2.waitKey(1) == 13 or img_id == 200:
                break
        cap.release()
        cv2.destroyAllWindows()
        messagebox.showinfo('Result', 'Generating dataset completed!!!')

    def generate_dataset(self):
        if self.mssv_entry.get() == "" or self.name_entry.get() == "" or self.age_entry.get() == "":
            messagebox.showinfo('Result', 'Please provide complete details of the user')
        else:
            user_id = self.mssv_entry.get()
            name = self.name_entry.get()
            age = self.age_entry.get()
            self.db.insert_user(user_id, name, age)
            threading.Thread(target=self.capture_faces).start()
