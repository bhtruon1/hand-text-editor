"""
Simply display the contents of the webcam with optional mirroring using OpenCV 
via the new Pythonic cv2 interface.  Press any key to quit.
"""
import time
import cv2
import numpy as np
from PIL import Image
import threading

images = []

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

def formatImage(img, count):
    #(h, w) = img.shape[:2]
    #r = 128 / float(w)
    #dsize = (128, int(h * r))
    #img = rgb2gray(img)
    #img = cv2.resize(img, dsize=dsize, interpolation=cv2.INTER_CUBIC)
    #img = cv2.GaussianBlur(img, (5, 5), 0)
    cv2.imwrite("images/image{}.jpeg".format(count), img)
    images.append(img)
    return img     

def show_webcam(mirror=False):
    cam = cv2.VideoCapture(0)
    l = np.zeros(10)
    start = time.time()
    count = 0
    while True:
        #ret_val is a boolan statement indicating that the value has been returned
        ret_val, img = cam.read()
        if mirror: 
            img = cv2.flip(img, 1)
        cv2.imshow('my webcam', img)
        #captures image every 3 seconds
        if (time.time() - start) % 60 >= 3:
            formatImage(img, count)  
            count += 1
            start = time.time()
        if cv2.waitKey(1) != -1: 
            break  # esc to quit
    cv2.destroyAllWindows()
    #print(images)

def main():
    show_webcam(mirror=True)


main()
