"""
Simply display the contents of the webcam with optional mirroring using OpenCV 
via the new Pythonic cv2 interface.  Press any key to quit.
"""
import torch
import torchvision
import torchvision.transforms as transforms
import time
import cv2
import os
import numpy as np
from PIL import Image
#from text_editor import *
from setGPU import *
from torch.autograd import Variable
classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I','K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T','U', 'V', 'W', 'X', 'Y']
classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T','U', 'V', 'W', 'X', 'Y', 'Z', 'BS', 'SP')

def checkImgDir():
    if not os.path.exists("images"):
        os.makedirs("images") 

def run_nn(model):
    text = ""
    img = Image.open("images/image.jpeg")
    ptLoader = transforms.Compose([transforms.ToTensor()]) 
    img = ptLoader( img ).float()
    img = Variable(img)
    img = img.unsqueeze(0)
    pred = model(img.cuda())
    values = pred.cpu().detach().numpy()
    pred = np.argmax(values[0])
    pred = classes[pred] 
    if pred == "SP":
       text += " "
    if pred == "BS":
       text += "BS"
    else:
       text += pred
    print(pred)
    print(text)
    print("")

def formatImage(img, width, maxwidth, model):
    cropwidth = int(width * 2)
    crop = img[0:width, cropwidth:maxwidth]
    crop = cv2.resize(crop, (32, 32))
    crop = cv2.Canny(crop, 100, 200)
    crop = cv2.cvtColor(crop, cv2.COLOR_GRAY2RGB)
    cv2.imwrite("images/image.jpeg", crop)
    run_nn(model)     

def show_webcam(model, mirror=False):
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
            formatImage(img, width, maxwidth, model)  
            start = time.time()

        if cv2.waitKey(1) != -1: 
            break  # esc to quit
    cv2.destroyAllWindows()

def main():
    checkImgDir()
    model = load_model('model.pth.tar') 
    show_webcam(model, mirror=True)

main()
