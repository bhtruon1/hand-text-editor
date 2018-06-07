import os
from PIL import Image
import sys
import cv2
import numpy as np
from matplotlib import pyplot as plt

size = 32, 32
mainDir = "data"

for subdir, dirs, files in os.walk(mainDir):
    for f in files:
        if ".JPG" in f or ".jpg" in f or ".jpeg" in f:
            try:
                path = os.path.join(subdir, f)
                img = cv2.imread(path, 0)
                edges = cv2.Canny(img,100,200)
                if not os.path.exists("edge" + subdir):
                    os.makedirs("edge" + subdir)
                cv2.imwrite( "edge" + path, edges)
            except IOError:
                print("cannot create thumbnail for {}".format(path))


