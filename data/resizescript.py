import os
from PIL import Image
import sys

size = 32, 32
mainDir = "Sign-Language-Digits-Dataset"

for subdir, dirs, files in os.walk(mainDir):
    for f in files:
        if ".JPG" in f or ".jpg" in f:
            try:
                path = os.path.join(subdir, f)
                im = Image.open(path)
                im.thumbnail(size, Image.ANTIALIAS)
                im.save(path, "JPEG")
            except IOError:
                print("cannot create thumbnail for {}".format(path))


