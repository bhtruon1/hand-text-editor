"""
Simply display the contents of the webcam with optional mirroring using OpenCV 
via the new Pythonic cv2 interface.  Press <esc> to quit.
"""

import cv2
import numpy as np
from PIL import Image
width = 32


def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

def formatImage(img):
    img = rgb2gray(img)
#    wpercent = width/float(img.shape[0])
#    hsize = int(float(img.shape[1])*float(wpercent))
    img = cv2.resize(img, dsize=(64, 64), interpolation=cv2.INTER_CUBIC)
    cv2.imwrite("image.jpeg", img)
    return img     

def show_webcam(mirror=False):
    cam = cv2.VideoCapture(0)
    l = np.zeros(10)
    while True:
        #ret_val is a boolan statement indicating that the value has been returned
        ret_val, img = cam.read()
        if mirror: 
            img = cv2.flip(img, 1)
        #print(np.array(rgb2gray(img)).shape)
        cv2.imshow('my webcam', img)
        formatImage(img)
        if cv2.waitKey(1) != -1: 
            break  # esc to quit
    cv2.destroyAllWindows()


def main():
    show_webcam(mirror=True)


main()
