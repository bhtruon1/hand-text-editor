"""
Simply display the contents of the webcam with optional mirroring using OpenCV 
via the new Pythonic cv2 interface.  Press <esc> to quit.
"""

import cv2
import numpy as np


def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

def show_webcam(mirror=False):
    cam = cv2.VideoCapture(0)
    l = np.zeros(10)
    while True:
        #ret_val is a boolan statement indicating that the value has been returned
        ret_val, img = cam.read()
        if mirror: 
            img = cv2.flip(img, 1)
        #print(np.array(rgb2gray(img)).shape)
        #cv2.imshow('my webcam', rgb2gray(img))
        print cv2.waitKey(1)
        if cv2.waitKey(1) == 27: 
            break  # esc to quit
    cv2.destroyAllWindows()


def main():
    show_webcam(mirror=True)


if __name__ == '__main__':
    main()
