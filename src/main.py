"""
Simply display the contents of the webcam with optional mirroring using OpenCV 
via the new Pythonic cv2 interface.  Press any key to quit.
"""
import time
import cv2
import os
import numpy as np
from PIL import Image
import threading
from text_editor import *
from setGPU import *

def run_nn(img, model, text):
    pred = model(img)
    if pred == "SP":
       text.insert(" ")
    if pred == "BS":
       text.delete()
    else:
       text.insert(pred) 

def formatImage(img, width, maxwidth, model, text):
    cropwidth = int(width * 2)
    crop = img[0:width, cropwidth:maxwidth]
    crop = cv2.resize(crop, (32, 32))
    crop = cv2.Canny(crop, 100, 200)
    run_nn(img, model, text)     

def show_webcam(text, model, mirror=False):
    cam = cv2.VideoCapture(0)
    maxwidth = int(cam.get(3))
    width = int(maxwidth/3)
    l = np.zeros(10)
    start = time.time()

    while True: 
        #ret_val is a boolan statement indicating that the value has been returned
        ret_val, img = cam.read()
        #crop the image to top corner of screen
        cv2.rectangle(img, (width, width), (0, 0), (0, 255, 0), 0)

        if mirror: 
            img = cv2.flip(img, 1)

        cv2.imshow('my webcam', img)

        #captures image every 5 seconds
        if (time.time() - start) % 60 >= 5:
            formatImage(img, width, maxwidth, model, text)  
            start = time.time()

        if cv2.waitKey(1) != -1: 
            break  # esc to quit
    cv2.destroyAllWindows()

def main():
    model = load_model('model.pth.tar') 
    text = TextEditor()
    show_webcam(text, model, mirror=True)


main()
