import os
from PIL import Image
import sys
import cv2

size = 32, 32
mainDir = "data"

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

for subdir, dirs, files in os.walk(mainDir):
    for f in files:
        if ".JPG" in f or ".jpg" in f or ".jpeg" in f:
            try:
                path = os.path.join(subdir, f)
                img = cv2.imread(path)
                img = cv2.cvtColor( img, cv2.COLOR_RGB2GRAY )
                if not os.path.exists("grey" + subdir):
                    os.makedirs("grey" + subdir)
                cv2.imwrite( "grey" + path, img )
            except IOError:
                print("cannot create thumbnail for {}".format(path))


