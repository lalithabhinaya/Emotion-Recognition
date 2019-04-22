import cv2 
import numpy as np
import random
import sys
import getopt
import os

def addNoise(image,prob):
    noiseOutput = np.zeros(image.shape,np.uint8)
    thres = 1 - prob
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = random.random()            
            if rdn < prob:
                noiseOutput[i][j] = 0
            elif rdn > thres:
                noiseOutput[i][j] = 255        
            else:
                noiseOutput[i][j] = image[i][j]                     	
    return noiseOutput	
 
def generateImages(img):
    noOfImages = 2
    randomizerLevel = 30
    outputFolder = "/home/CAP5627-3/Augmented_images/No_pain/"
    noiseLevel = 0.05
    borderMode = cv2.BORDER_CONSTANT#cv2.BORDER_REPLICATE
    local = str(img)
    img_new = cv2.imread(local, cv2.IMREAD_UNCHANGED)
    rows, cols = img_new.shape
    
    fileNm=os.path.basename(img)
    fileName=os.path.splitext(fileNm)[0]
    originalImage = cv2.imread(img)
    outputFile = outputFolder+fileNm
    cv2.imwrite(outputFile,originalImage)
    for i in range(0,noOfImages):
        image=img_new
        randomNumber = random.randint(2,randomizerLevel)
        randomSign = 1
        if(random.randint(1,2) == 2):
            randomSign = (randomSign*-1)	
        if(noiseLevel != 0.0 ):
            image = addNoise(image,random.uniform(0, noiseLevel))	
        #Rotation 
        M = cv2.getRotationMatrix2D((cols/2,rows/2),randomizerLevel * randomNumber ,1)
        image = cv2.warpAffine(image,M,(cols,rows),borderMode=borderMode)		
        #Translation
        M = np.float32([[random.uniform(0.5, 1),0,cols%randomNumber* randomSign] ,[0,random.uniform(0.5, 1) ,rows%randomNumber * randomSign]])
        image = cv2.warpAffine(image,M,(cols,rows),borderMode=borderMode) 
        cv2.imwrite(''.join([outputFolder,fileName,"_",str(i), ".jpg"]),image)
        i=i+1
        
    
    
    
    
def fetchImages(path):		
    for data in os.scandir(path):	
        #for pain in os.scandir(data.path)
        generateImages(data.path)   
	

def main():
    datapath = sys.argv[1]
    print(datapath)
    fetchImages(datapath)
main()
