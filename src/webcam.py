"""
Simply display the contents of the webcam with optional mirroring using OpenCV 
via the new Pythonic cv2 interface.  Press any key to quit.
"""
import time
import cv2
import numpy as np
from PIL import Image
import threading

allimages = True
counter = True
images = []

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

def checkContours(crop):
   # grey = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
   # blurred = cv2.GaussianBlur(grey, (35, 35), 0)
   # _, thresh = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
   # contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
   # cmax = max(contours, key=lambda x: cv2.contourArea(x))
   # x, y, w, h = cv2.boundingRect(cmax)
   # if x == 0 and y == 0:
   #     return False
    return True

def formatImage(img, width, maxwidth, count):
    cropwidth = int(width * 2)
    crop = img[0:width, cropwidth:maxwidth]
    if checkContours(crop):
        crop = cv2.resize(crop, (32, 32))
        if counter:
            cv2.imwrite("images/image{}.jpeg".format(count), crop)
        else:
            cv2.imwrite("images/image.jpeg".format(count), crop)
        images.append(img)
    return img     

def show_webcam(mirror=False):
    cam = cv2.VideoCapture(0)
    maxwidth = int(cam.get(3))
    width = int(maxwidth/3)
    l = np.zeros(10)
    start = time.time()
    count = 0

    while True: 
        #ret_val is a boolan statement indicating that the value has been returned
        ret_val, img = cam.read()
        #crop the image to top corner of screen
        cv2.rectangle(img, (width, width), (0, 0), (0, 255, 0), 0)

        if mirror: 
            img = cv2.flip(img, 1)

        cv2.imshow('my webcam', img)

        #captures image every 3 seconds
        if (time.time() - start) % 60 >= 5 or allimages:
            formatImage(img, width, maxwidth, count)  
            count += 1
            start = time.time()

        if cv2.waitKey(1) != -1: 
            break  # esc to quit
    cv2.destroyAllWindows()
    #print(images)

def main():
    show_webcam(mirror=True)


main()
