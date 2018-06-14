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

allimages = True
grey = False
edge = False
counter = True
max_count = 108
seconds = .5
images = []
label = 0

def checkImgDir():
    if not os.path.exists("/Users/chenmo/Files/PythonProjects/local_cs175/new/" + str(label)):
        os.makedirs("/Users/chenmo/Files/PythonProjects/local_cs175/new/" + str(label))

def formatImage(img, width, maxwidth, count):
    cropwidth = int(width * 2)
    crop = img[0:width, cropwidth:maxwidth]
    crop = cv2.resize(crop, (64, 64))
    if grey:
        crop = cv2.cvtColor( img, cv2.COLOR_RGB2GRAY )
    if edge: 
        crop = cv2.Canny(crop, 100, 200)
    cv2.imwrite("/Users/chenmo/Files/PythonProjects/local_cs175/new/{}/{}_IMG_{:04}.jpeg".format(str(label), str(label), count), crop)
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

        #captures image every 0.n seconds
        if (time.time() - start) % 60 >=  seconds:
            formatImage(img, width, maxwidth, count)  
            count += 1
            start = time.time()

        if cv2.waitKey(1) != -1 or count == max_count: 
            break  # esc to quit
    cv2.destroyAllWindows()
    #print(images)

def main():
    checkImgDir()
    show_webcam(mirror=True)


main()
