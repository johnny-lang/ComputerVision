import tkinter as tk
from video_feed import VideoFeed
from dataset_generator import DatasetGenerator
from classifier_trainer import ClassifierTrainer

def main():
    # Create the main window
    window = tk.Tk()
    window.title("Face Recognition System")
    window.geometry("1200x800")

    # Create the label frame for video feed
    label_frame = tk.Label(window, bg='white')
    label_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    # Create the control frame for buttons and entries
    control_frame = tk.Frame(window, bg='lightgray')
    control_frame.pack(side=tk.RIGHT, fill=tk.Y)

    # Create entry labels and fields
    labels = ["MSSV", "Tên", "Tuổi"]
    entries = []
    for i, text in enumerate(labels):
        label = tk.Label(control_frame, text=text, font=("Algerian", 12))
        label.grid(column=0, row=i)
        entry = tk.Entry(control_frame, width=30, bd=5)
        entry.grid(column=1, row=i)
        entries.append(entry)

    mssv_entry, name_entry, age_entry = entries

    # Create buttons and attach handlers
    trainer = ClassifierTrainer()
    tk.Button(control_frame, text="Training", font=("Algerian", 12), bg='orange', fg='red', command=trainer.train).grid(column=1, row=4)

    video_feed = VideoFeed(label_frame)
    tk.Button(control_frame, text="Detect Face", font=("Algerian", 12), bg='green', fg='white', command=video_feed.toggle_video).grid(column=1, row=5)

    generator = DatasetGenerator(mssv_entry, name_entry, age_entry, label_frame)
    tk.Button(control_frame, text="Generate dataset", font=("Algerian", 12), bg='pink', fg='black', command=generator.generate_dataset).grid(column=1, row=6)

    # Start the main loop
    video_feed.start_video()
    window.mainloop()

if __name__ == "__main__":
    main()
