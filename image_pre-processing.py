import numpy as np
import cv2
import os
import sys
from PIL import Image


def doCropImages(path):
    face_cas = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    for dataset in os.scandir(path):
      for types in os.scandir(dataset.path):
        print(types.name)
        for labels in os.scandir(types.path):
          for f in os.scandir(labels.path):
              source = f.path
              image = cv2.imread(source, cv2.IMREAD_COLOR)
              #gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
              face = face_cas.detectMultiScale(image, 1.3, 5)
              for (x,y,w,h) in face:
                  avg = max(w, h)/2
                  cX = x + w/2
                  cY = y + h/2
                  fX = int(cX - avg)
                  fY = int(cY - avg)
                  #fAvg = int(2*avg)
                  img = image[fY:fY+160, fX:fX+160]
                  clahe = cv2.createCLAHE(clipLimit=9, tileGridSize=(8,8))    
                  cl = ''
                  normalizedImg = np.zeros((150, 150))
                  normalizedImg = cv2.normalize(img,  normalizedImg, 0, 255, cv2.NORM_MINMAX)    
                  #smoothened = cv2.GaussianBlur(normalizedImg,(3,3),cv2.BORDER_DEFAULT)
                  if(np.size(normalizedImg) > 5):
                    gray = cv2.cvtColor(normalizedImg, cv2.COLOR_BGR2GRAY)
                    cl = clahe.apply(gray)

                  destination = ''
                  if(types.name == 'Training' and labels.name == 'Pain'):
                    destination = '/home/CAP5627-3/test_folder/Training/Pain/'
                    destination = destination+f.name
                  if(types.name == 'Training' and labels.name == 'No_pain'):
                    destination = '/home/CAP5627-3/test_folder/Training/No_pain/'
                    destination = destination+f.name
                  if(types.name == 'Testing' and labels.name == 'Pain'):
                    destination = '/home/CAP5627-3/test_folder/Testing/Pain/'
                    destination = destination+f.name
                  if(types.name == 'Testing' and labels.name == 'No_pain'):
                    destination = '/home/CAP5627-3/test_folder/Testing/No_pain/'
                    destination = destination+f.name
                  if(types.name == 'Validaiton' and labels.name == 'Pain'):
                    destination = '/home/CAP5627-3/test_folder/Validation/Pain/'
                    destination = destination+f.name
                  if(types.name == 'Validaiton' and labels.name == 'No_pain'):
                    destination = '/home/CAP5627-3/test_folder/Validation/No_pain/'
                    destination = destination+f.name
                  if(np.size(normalizedImg) > 5):
                    cv2.imwrite(destination, cl)



def main():
    param = '/home/CAP5627-3/new_input/'
    doCropImages(param)
            
main()