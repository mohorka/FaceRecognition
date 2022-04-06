from random import randint
from tkinter import *
from tkinter import filedialog
from tkinter.ttk import Combobox

import cv2 as cv
from PIL import Image, ImageTk
from sklearn.datasets import fetch_olivetti_faces


def download_photo():
    global image_path
    global is_randomly
    is_randomly = 0
    image_path = filedialog.askopenfilename()


def download_template():
    global template_path
    template_path = filedialog.askopenfilename()

def delete():
    label.destroy()
    label2.destroy()
    text1.destroy()
    text2.destroy()

def get_from_dataset():
    global is_randomly
    global rand_image
    global rand_template
    is_randomly = 1
    faces = fetch_olivetti_faces()
    faces = faces.images
    img_idx = randint(0, len(faces) - 1)
    temp_idx = randint(0, len(faces) - 1)
    rand_image = faces[img_idx] * 255
    rand_template = faces[temp_idx] * 255
    rand_template = rand_template[10:50, 10:50]


def matchingTemplateDetector():
    global label
    global label2
    global text1
    global text2
    m = method_name.get()
    if is_randomly == 1:
        image = rand_image.copy()
        template = rand_template.copy()
    else:
        image = cv.imread(image_path, 0)
        template = cv.imread(template_path, 0)
    method = eval("cv." + m)
    w, h = template.shape[::-1]
    w_im, h_im = image.shape[::-1]
    
    if w_im > w:
        new_w = w_im // 1
        new_h = h_im // 1
        image = cv.resize(image,dsize=(new_w, new_h),interpolation=cv.INTER_CUBIC)
    
    res = cv.matchTemplate(image, template, method)
    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)

    if method in [cv.TM_SQDIFF, cv.TM_SQDIFF_NORMED]:
        top_left = min_loc
    else:
        top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)

    image = cv.rectangle(image, top_left, bottom_right, 255, 2)

    image = Image.fromarray(image)
    template = Image.fromarray(template)

    width, height = image.size
    widthtemp, heighttemp = template.size

    #if width >= 400 or height >= 400:
        #image = image.resize((width // 3, height // 3), Image.ANTIALIAS)

    #if is_randomly == 0:
        #image = image.resize((width // 3, height // 3), Image.ANTIALIAS)
        #template = template.resize((widthtemp // 3, heighttemp // 3), Image.ANTIALIAS)
    #else:
    if is_randomly == 1:
        image = image.resize((150, 150), Image.ANTIALIAS)
        template = template.resize((150, 150), Image.ANTIALIAS)

    text1 = Label(window, text="Result")
    text1.grid(row=3, column=1)
    text2 = Label(window, text="Template")
    text2.grid(row=3, column=3)

    photo = ImageTk.PhotoImage(image)
    label = Label(window, image=photo)
    label.image = photo
    label.grid(row=4, column=1)

    temp_photo = ImageTk.PhotoImage(template)
    label2 = Label(window, image=temp_photo)
    label2.image = temp_photo
    label2.grid(row=4, column=3)


window = Tk()
window.title("Template Matching Detector")
window.geometry("1200x600")

lbl = Label(window, text="Choose method")
lbl.grid(column=0, row=0)

method_name = StringVar()


combo = Combobox(window, textvariable=method_name)
combo["values"] = (
    "TM_CCOEFF",
    "TM_CCOEFF_NORMED",
    "TM_CCORR",
    "TM_CCORR_NORMED",
    "TM_SQDIFF",
    "TM_SQDIFF_NORMED",
)
combo.current(0)
combo.grid(column=0, row=1)

button1 = Button(window, text="Download photo", command=download_photo, width=20)
button1.grid(column=1, row=1)

button2 = Button(window, text="Download template", command=download_template, width=20)
button2.grid(column=2, row=1)

button3 = Button(window, text="Choose randomly", command=get_from_dataset, width=20)
button3.grid(column=3, row=1)

button = Button(window, text="Go", command=matchingTemplateDetector, width=20)
button.grid(column=0, row=2)

button = Button(window, text="Clear", command=delete, width=20)
button.grid(column=0, row=3)

window.mainloop()
