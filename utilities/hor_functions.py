import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import os
import utilities.swt as swt
import imutils

from collections import defaultdict
import hashlib
from typing import TypeVar, NamedTuple, List, Optional, Tuple
import math
import scipy.ndimage as ndimage
import sys
from scipy.spatial import Voronoi, voronoi_plot_2d
import scipy.sparse, scipy.spatial
from scipy.signal import find_peaks
import time


diagnostics = False





class utility:   
    def read_img(self,img):
        try:
            image = cv2.imread('input_images/horizontal/'+img,)
            plt.imshow(image,cmap='gray')
            plt.show()
            return image
        except:
            print("Image does not exist.")
            sys.exit()
            
    def preprocessing(self,image):
        # Grayscale 
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
        # blur
        blur = cv2.GaussianBlur(gray,(5,5),0)
        # adaptive threshold
        self.th5 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
        # erosion
        self.kernel = np.ones((3,3), np.uint8) 
        self.img_erosion = cv2.erode(self.th5, self.kernel, iterations=1)
        plt.imshow(self.img_erosion,cmap='gray')
        plt.show()
        return image
    
    
    def stroke_width_tranform(self,image):
        output=swt.main(image)
        
        plt.imshow(output)
        plt.show()
        ret,thresh1 = cv2.threshold(image,127,255,cv2.THRESH_BINARY)
        
        print(output.shape,image.shape)
        print(type(image))
        out,im=0,0
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                out= out+output[i,j]
                im = im+ thresh1[i,j]
                 
        print(out,im)
        awg_width= out/im;
        print(awg_width)
       
       

    # plt.title('Vertical Projection')
    # plt.plot(sum_x)
    # plt.xlim([0, height])
    # plt.show()
    # plt.title('Skew Image')
    # plt.imshow(img)
    # plt.show()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray,(5,5),0)
        # adaptive threshold
        th5 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
        # erosion
        
        plt.imshow(th5)
        plt.show()
        img1 = swt.skew_correct(th5)
        img2 = swt.vertical_proj(img1)
        sum_x = swt.horizontal_proj(img2)
        blur = cv2.GaussianBlur(image,(5,5),0)

        height, width = th5.shape[:2]

        cordin=[]
        line=list()
        count=0
        j=0
#thresh = 0.05*max(sum_x[0])
#thresh = (np.mean(minima)-np.std(minima))
        
        for i in range(len(sum_x[0])):
            line.append(count)
            count=[]
            if i<th5.shape[1] and sum_x[0][i]<=awg_width.any():
        #count.append()
                cv2.line(th5,(j,0),(j,th5.shape[0]),(0,0,60))
        #i+=1
                cordin.append(j)

            j+=1

# plt.plot(sum_x[0])
# plt.xlim([0, width])
# plt.show()
        plt.imshow(th5,cmap='gray')
        plt.show()
        letter=[]
        kernel = np.ones((3,3), np.uint8) 
        for i in range(0,len(cordin)-1):
            if cordin[i]==cordin[i+1]-1:
                continue
            else:
                letter.append(cordin[i])
        letter1=[]
        for i in range(0,len(cordin)-1):
            if cordin[i-1]==cordin[i]-1:
                continue
            else:
                letter1.append(cordin[i])
        del letter1[0]

        img_lst=[]
        ccc=[]
        for i in range(len(letter1)):
            if (letter1[i]-letter[i]>10):
                img_lst.append(img2[:,letter[i]:letter1[i]])
                b=letter1[i]-letter[i]
                ccc.append(b)
                print(b)
        awg=ccc/len(img_lst)
        print(awg)       
               

        
        
        """points = [] 
        
        print(points)

        vor = Voronoi(points)

        fig = plt.figure(figsize=(20,20))
        ax = fig.add_subplot(111)
        ax.imshow(ndimage.rotate(thresh1, 90))
        voronoi_plot_2d(vor, point_size=10, ax=ax)
        plt.show()"""

        

        for i in img_lst:
            black = np.zeros((i.shape[1],1),np.uint8)
            black = cv2.bitwise_not(black)
            i = cv2.rotate(i, cv2.ROTATE_90_COUNTERCLOCKWISE)
            #blur = cv2.bilateralFilter(gray,9,75,75)
            blur = cv2.medianBlur(i, 3)
            ret,th4 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    #th3 = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
            
            res = cv2.matchTemplate(black,i,cv2.TM_SQDIFF_NORMED)
            for j in range(res.shape[1]):
                res[0][j] = round(res[0][j],2)
            lst=[]
            for j in range(len(res[0])):
                if res[0][j]!=0:
                    lst.append(j)
            img = th4[:,min(lst):max(lst)]
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            plt.imshow(img,cmap='gray')
            plt.show()
       



         

   
      
           