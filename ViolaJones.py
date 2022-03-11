from tkinter import *
from tkinter import filedialog
from tkinter.ttk import Combobox

import cv2 as cv
from PIL import Image, ImageTk


def download_photo():
    global image_path
    image_path = filedialog.askopenfilename()


def ViolaJones():
    image = cv.imread(image_path)
    grayscale_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    face_cascade = cv.CascadeClassifier(
        cv.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    eye_cascade = cv.CascadeClassifier(cv.data.haarcascades + "haarcascade_eye.xml")
    detected_faces = face_cascade.detectMultiScale(grayscale_image)
    detected_eyes = eye_cascade.detectMultiScale(grayscale_image)
    for (column, row, width, height) in detected_faces:
        cv.rectangle(
            image, (column, row), (column + width, row + height), (255, 0, 0), 4
        )
    for (column, row, width, height) in detected_eyes:
        cv.rectangle(
            image, (column, row), (column + width, row + height), (0, 255, 0), 4
        )

    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    width, height = image.size
    image = image.resize((width // 3, height // 3), Image.ANTIALIAS)
    Label(window, text="Result").grid(row=1, column=3)
    photo = ImageTk.PhotoImage(image)
    label = Label(window, image=photo)
    label.image = photo
    label.grid(row=2, column=3)


window = Tk()
window.title("Viola Jones Detector")
window.geometry("800x400")

button = Button(window, text="Download photo", command=download_photo, width=20)
button.grid(column=0, row=0)

button = Button(window, text="Go", command=ViolaJones, width=20)
button.grid(column=0, row=1)
window.mainloop()
