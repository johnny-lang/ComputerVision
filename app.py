import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import cv2
import os
import numpy as np
import mysql.connector
import threading

# Create the main window
window = tk.Tk()
window.title("Face Recognition System")

window_width = 1200
window_height = 800
window.geometry(f"{window_width}x{window_height}")

# Create the label frame for video feed
label_frame = tk.Label(window, bg='white')
label_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

# Create the control frame for buttons and entries
control_frame = tk.Frame(window, bg='lightgray')
control_frame.pack(side=tk.RIGHT, fill=tk.Y)

l1 = tk.Label(control_frame, text="MSSV", font=("Algerian", 12))
l1.grid(column=0, row=0)
t1 = tk.Entry(control_frame, width=30, bd=5)
t1.grid(column=1, row=0)

l2 = tk.Label(control_frame, text="Tên", font=("Algerian", 12))
l2.grid(column=0, row=1)
t2 = tk.Entry(control_frame, width=30, bd=5)
t2.grid(column=1, row=1)

l3 = tk.Label(control_frame, text="Tuổi", font=("Algerian", 12))
l3.grid(column=0, row=2)
t3 = tk.Entry(control_frame, width=30, bd=5)
t3.grid(column=1, row=2)

video_running = True  # Flag to control video feed

def train_classifier():
    data_dir = "D:\Face_detection\data"
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

    # Train the classifier and save
    clf = cv2.face.LBPHFaceRecognizer_create()
    clf.train(faces, ids)
    clf.write("classifier.xml")
    messagebox.showinfo('Result', 'Training dataset completed!!!')

b1 = tk.Button(control_frame, text="Training", font=("Algerian", 12), bg='orange', fg='red', command=train_classifier)
b1.grid(column=1, row=4)

def detect_face():
    global video_running

    if not video_running:
        video_running = True
        start_video()
    else:
        video_running = False

import cv2
import mysql.connector
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os
from tkinter import messagebox
from tkinter import Label
import tkinter as tk
from PIL import ImageTk

def start_video():
    def draw_boundary(img, classifier, scaleFactor, minNeighbors, clf, mycursor):
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        features = classifier.detectMultiScale(gray_image, scaleFactor, minNeighbors)

        coords = []

        for (x, y, w, h) in features:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), 2)
            id, pred = clf.predict(gray_image[y:y + h, x:x + w])

            try:
                confidence = int(100 * (1 - pred / 300))
            except:
                confidence = 0
                
            mycursor.execute("SELECT name FROM dang_ki WHERE id=%s", (id,))
            s = mycursor.fetchone()

            if s and confidence > 70:
                name = s[0]  # Assuming 'name' is the first field in the table
                name = name.encode('utf-8').decode('utf-8')  # Decode correctly
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # Convert OpenCV image to PIL image
                img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                draw = ImageDraw.Draw(img_pil)
                font = ImageFont.truetype("D:\Face_detection\Merriweather-Regular.ttf", 32)  # Update the path to your font
                draw.text((x, y - 30), name, font=font, fill=(0, 255, 0))
            else:
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(img, f'{id}', (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 1, cv2.LINE_AA)

            coords = [x, y, w, h]
        return coords

    def recognize(img, clf, faceCascade, mycursor):
        coords = draw_boundary(img, faceCascade, 1.1, 10, clf, mycursor)
        return img

    faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    clf = cv2.face.LBPHFaceRecognizer_create()

    # Load classifier if it exists
    if os.path.exists("classifier.xml"):
        clf.read("classifier.xml")
    else:
        clf = None
    
    try:
        mydb = mysql.connector.connect(
            host="localhost",
            user="root",
            passwd="Khanhminh04-12",
            database="users",
            charset='utf8mb4',  # Use utf8mb4 to ensure full Unicode support
            use_unicode=True
        )
    except mysql.connector.Error as err: 
        messagebox.showerror('Database Error', f"Error connecting to MySQL: {err}")
    

    video_capture = cv2.VideoCapture(0)
    video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
    video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)

    def update_frame():
        global video_running
        mycursor = mydb.cursor()
        if video_running:
            ret, frame = video_capture.read()
            if ret:
                frame = cv2.flip(frame, 1)
                if clf:
                    frame = recognize(frame, clf, faceCascade, mycursor)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame)
                imgtk = ImageTk.PhotoImage(image=img)
                label_frame.imgtk = imgtk
                label_frame.configure(image=imgtk)
            label_frame.after(10, update_frame)
        else:
            video_capture.release()
            label_frame.configure(image='')

    update_frame()


b2 = tk.Button(control_frame, text="Detect Face", font=("Algerian", 12), bg='green', fg='white', command=detect_face)
b2.grid(column=1, row=5)

def generate_dataset():
    if t1.get() == "" or t2.get() == "" or t3.get() == "":
        messagebox.showinfo('Result', 'Please provide complete details of the user')
    else:
        id = t1.get()
        name = t2.get()
        age = t3.get()

        # Connect to MySQL database
        try:
            mydb = mysql.connector.connect(
                host="localhost",
                user="root",
                passwd="Khanhminh04-12",
                database="users",
                port=3306,
            )
        except mysql.connector.Error as err:
            messagebox.showerror('Database Error', f"Error connecting to MySQL: {err}")
            return
        
        mycursor = mydb.cursor()
        # Insert data into the database
        insert_query = "INSERT INTO dang_ki (id, name, age) VALUES (%s, %s, %s)"
        val = (id, name, age)
        mycursor.execute(insert_query, val)
        mydb.commit()
 
        face_classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

        def face_cropped(img, face_coords=None):
            if face_coords is None:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = face_classifier.detectMultiScale(gray, 1.3, 5)
                if len(faces) == 0:
                    return None, None
                x, y, w, h = faces[0]
                return img[y:y + h, x:x + w], (x, y, w, h)
            else:
                x, y, w, h = face_coords
                return img[y:y + h, x:x + w], face_coords
            
            
        def capture_faces():
            cap = cv2.VideoCapture(0)
            orientations = [
                ("front", "Look straight at the camera", 25),
                ("left", "Turn your face to the left", 25),
                ("right", "Turn your face to the right", 25),
                ("up", "Tilt your head up", 25),
                ("down", "Tilt your head down", 25),
            ]
            img_id = 0
            face_coords = None

            for orientation, message, count in orientations:
                messagebox.showinfo('Orientation', message)
                captured = 0
                while captured < count:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    cropped_face, face_coords = face_cropped(frame, face_coords)
                    if cropped_face is not None:
                        img_id += 1
                        captured += 1
                        face = cv2.resize(cropped_face, (120, 200))
                        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                        file_name_path = f"D:/Face_detection/data/user.{id}.{img_id}.{orientation}.jpg"
                        cv2.imwrite(file_name_path, face)
                        cv2.putText(face, str(img_id), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

                        # Update the GUI with the captured face
                        face_rgb = cv2.cvtColor(face, cv2.COLOR_GRAY2RGB)
                        img = Image.fromarray(face_rgb)
                        img = img.resize((300, 400), Image.LANCZOS)
                        imgtk = ImageTk.PhotoImage(image=img)
                        label_frame.imgtk = imgtk
                        label_frame.configure(image=imgtk)

                if cv2.waitKey(1) == 13:
                    break
            cap.release()
            cv2.destroyAllWindows()
            messagebox.showinfo('Result', 'Generating dataset completed!!!')

        # Run the face capture in a separate thread
        threading.Thread(target=capture_faces).start()

b3 = tk.Button(control_frame, text="Generate dataset", font=("Algerian", 12), bg='pink', fg='black', command=generate_dataset)
b3.grid(column=1, row=6)

def main():
    start_video()  
    window.mainloop()

if __name__ == "__main__":
    main()
