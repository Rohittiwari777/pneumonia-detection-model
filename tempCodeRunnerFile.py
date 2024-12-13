import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, Label, Button, Canvas

# Load the trained model
model = load_model('xray_classification_model.h5')

# Function to preprocess the image
def preprocess_image(file_path):
    try:
        img = cv2.imread(file_path)
        if img is None:
            raise ValueError("Invalid Image")
        img_resized = cv2.resize(img, (150, 150))
        img_normalized = img_resized / 255.0
        return np.expand_dims(img_normalized, axis=0)
    except Exception as e:
        print(f"Error: {e}")
        return None

# Function to predict the class
def predict_image():
    file_path = filedialog.askopenfilename()
    img = preprocess_image(file_path)
    if img is None:
        result_label.config(text="Improper Image. Please select a valid X-ray image.", fg="red")
    else:
        prediction = model.predict(img)
        class_names = ['Normal', 'Pneumonia']
        predicted_class = class_names[np.argmax(prediction)]
        confidence = np.max(prediction) * 100
        result_label.config(text=f"Prediction: {predicted_class} ({confidence:.2f}%)", fg="green")

# Create GUI
root = tk.Tk()
root.title("Pneumonia Detection System")
root.geometry("600x400")
root.configure(bg="#f0f0f5")

# Background Design
canvas = Canvas(root, width=600, height=400, bg="#e6f7ff")
canvas.pack(fill="both", expand=True)
canvas.create_rectangle(50, 50, 550, 350, outline="#33a1c9", width=3)
canvas.create_text(300, 80, text="Pneumonia Detection System", font=("Helvetica", 24, "bold"), fill="#004d80")

# Buttons and Labels
select_button = Button(root, text="Select X-ray Image", command=predict_image, font=("Helvetica", 14), 
                       bg="#4CAF50", fg="white", activebackground="#45a049", padx=20, pady=10)
select_button_window = canvas.create_window(300, 180, window=select_button)

result_label = Label(root, text="", font=("Helvetica", 16), bg="#e6f7ff")
result_label_window = canvas.create_window(300, 260, window=result_label)

root.mainloop()
