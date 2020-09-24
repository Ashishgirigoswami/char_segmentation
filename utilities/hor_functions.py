import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import os
import utilities.swt as swt
import utilities.thinig_algo as thin
import utilities.path as distance
import utilities.shortest as short
import utilities.thinning as thinning
from PIL import Image
import imutils
import math
from .id_colors import build_colormap
from skimage.morphology import skeletonize
from skimage.feature import peak_local_max
import statistics 
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
    #Read file function
    def read_img(self,img):
        try:
            image = cv2.imread('input_images/horizontal/'+img,)
            plt.imshow(image,cmap='gray')
            plt.show()
            return image
        except:
            print("Image does not exist.")
            sys.exit()
    #preprocessing function        
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
    
    #stroke width tranform function
    def stroke_width_tranform(self,image):
        #runnning swt algo and taking output of swt
        output=swt.main(image)
        plt.imshow(output)
        plt.show()
        #blur origgnal image
        blur = cv2.GaussianBlur(image,(3,3),0)
        ret,thresh1 = cv2.threshold(blur,127,255,cv2.THRESH_BINARY)

        
        #finding average width by divinding each pixel of swt output with binarized image.
        out,im=0,0
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                out= out+output[i,j]
                im = im+ thresh1[i,j]
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
        blur = cv2.GaussianBlur(gray,(3,3),0)
        # adaptive threshold
        th5 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
        th6=th5.copy()
        
        plt.imshow(th5)
        plt.show()
        #finding vertical projection of of threshold image
        img1 = swt.skew_correct(th5)
        img2 = swt.vertical_proj(img1)
        sum_x = swt.horizontal_proj(img2)
      

        height, width = th5.shape[:2]

        cordin=[]
        line=list()
        count=0
        j=0
#thresh = 0.05*max(sum_x[0])
#thresh = (np.mean(minima)-np.std(minima))
        #Segmenting chachters where  projection value is zero
        for i in range(len(sum_x[0])):
            line.append(count)
            count=[]
            if i<image.shape[1] and sum_x[0][i]==0:
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
        #finding starting and ending part of each charchter segementation
        
        letter=[]
        letter_start=[]
        kernel = np.ones((3,3), np.uint8) 
        for i in range(0,len(cordin)-1):
            if cordin[i]==cordin[i-1]+1:
                continue
            else:
                letter.append(cordin[i])
        letter1=[]
        for i in range(0,len(cordin)-1):
            if cordin[i-1]==cordin[i]-1:
                continue
            else:
                letter1.append(cordin[i])
                letter_start.append(cordin[i])
        for i in range(len(letter)-1):
            if(letter1[i]-letter[i]<2):
                cordin.remove(letter1[i])
                if letter[i] in cordin: 
                    cordin.remove(letter[i])
              
        del letter1[0]

        img_lst=[]
        ccc=[]
        c=0
        bb=[]
        img_lst=[]
        print(letter1)
        print(letter)
        
        for i in range(len(letter1)):
            #if (letter1[i]-letter[i]>10):
            img_lst.append(img1[:,letter[i]:letter1[i]])
            b=letter1[i]-letter[i]
            ccc.append(b)
            c=c+b;
            bb.append(b)
        print(len(img_lst))
        print("2")
                
       
               
      
        if len(ccc)>1:
            print("inside")
            std_of_seg= np.std(ccc)
            mean= np.mean(ccc)
        else:
            print("INSIDE")
            std_of_seg=0
            mean=image.shape[1]
        
        max_width=round(std_of_seg+mean)+30
        min_width=round(mean-std_of_seg)
        print(mean,max_width,min_width)
       
        for k in img_lst:
            
            if (len(img_lst)==1):
                print("enter")
                for i in range(len(sum_x[0])):
                
                    
                    if i<k.shape[1] and sum_x[0][i]<=awg_width.min():
        #count.append()
                        cv2.line(k,(i,0),(i,k.shape[0]),(0,0,60))
                        cordin.append(i)

            else:
                if k.shape[1]>=max_width:
                    print(k.shape[1])
                    img1 = swt.skew_correct(k)
                    img2 = swt.vertical_proj(img1)
                    sum_x = swt.horizontal_proj(img2)
                    for i in range(len(sum_x[0])):
                        line.append(count)
                
                
                        if i<k.shape[1] and sum_x[0][i]<=awg_width.min():
        #count.append()
                            cv2.line(k,(i,0),(i,k.shape[0]),(0,0,60))
                            cordin.append(i)
        #i+=1
# plt.plot(sum_x[0])
# plt.xlim([0, width])
# plt.show()
        cordin = list(dict.fromkeys(cordin))
        cordin.sort(reverse=False)
        print(cordin)
        plt.imshow(k,cmap='gray')
        plt.show()
        letter=[]
        kernel = np.ones((3,3), np.uint8) 
        for i in range(0,len(cordin)-1):
            if cordin[i]==cordin[i-1]+1:
                continue
            else:
                letter.append(cordin[i])
        letter1=[]
        for i in range(0,len(cordin)-1):
            if cordin[i+1]==cordin[i]+1:
                continue
            else:
                letter1.append(cordin[i])
            # for i in range(len(letter)-1):
            #     if(letter1[i]-letter[i]<3):
            #         cordin.remove(letter1[i])
            #         if letter[i] in cordin: 
            #             cordin.remove(letter[i])
        print(letter)
        print(letter1)   
        del letter[0]

        img_lst=[]
        ccc=[]
        c=0
        bb=[]
        
        for i in range(len(letter1)):
                #if (letter1[i]-letter[i]>10):
            img_lst.append(img1[:,letter1[i]:letter[i]])
            b=letter[i]-letter1[i]
            ccc.append(b)
            c=c+b;
            bb.append(b)
                
        print(len(img_lst))
               
            
        std_of_seg= np.std(ccc)
        mean= np.mean(ccc)
        print(mean,std_of_seg)
       
        min_width=round(mean-std_of_seg)
        max_width=round(mean+min_width)
        print(mean,max_width,min_width)
            
            
            
            

        final_image_list=[]
        for i in img_lst:
            if i.shape[1]<max_width:
                final_image_list.append(i)
            if i.shape[1]>=max_width:
                plt.imshow(i)
                plt.show()
                print(i[0,0])
                #labels, components = connected_components(swt) 
                
                img11=thinning.main(i)
               
                img = Image.fromarray((255 * img11).astype("uint8")).convert("RGB")
                img=np.asarray(img)

                img12=cv2.bitwise_not(img)
                img13=cv2.bitwise_not(img12)
                print(img.shape)
                plt.imshow(img)
                plt.show()
                gray = cv2.cvtColor(img12, cv2.COLOR_BGR2GRAY)
                kernel = np.ones((3,3), np.uint8) 
                img_erosion = cv2.erode(gray, kernel, iterations=1) 

               
                #img11=cv2.bitwise_not(img11)
                 
        # blur
                
                
                
                ximg1 = swt.skew_correct(i)
                ximg2 = swt.vertical_proj(ximg1)
                xsum_x = swt.horizontal_proj(ximg2)
                xsum_x=swt.smooth(xsum_x[0],1)
               
                #mean= mean-std_of_seg
                peak, _ = find_peaks(xsum_x,distance=mean/2)
                plt.plot(xsum_x)
                plt.plot(peak, xsum_x[peak], "x")
                plt.show()
                print(peak)
                img14=cv2.bitwise_not(img13)
                
                for l in peak: 
                    print(l)      
                    cv2.line(img13,(l,0),(l,img.shape[0]),(255,255,255))
                    
                plt.imshow(img13,cmap='gray')
                plt.show()
                hh=int(img13.shape[0]/2)
                print(hh)
                path=0
                points = []
                while(path!=1):
                    path=distance.main(img13,img12,peak[0],peak[1],hh,hh)
                    print(path)
                    if(path==1):
                        mid=int(round(img12.shape[1]/2))
                        center=[0,mid]
                        break
                    mid=int(round(len(path)/2))+1
                    print(mid)
                    print(path[mid])
                    center= path[mid]
                    print(center)
                    points.append(center)
                    img13[center[0],center[1]]=[0,0,0]
                    img12[center[0],center[1]]=[255,255,255]
                    img14[center[0],center[1]]=[255,255,255]

                    plt.imshow(img13,cmap='gray')
                    plt.show()
                
                sum_b=0
                
                path_short=short.main(img14,center[1],center[1],0,img12.shape[0])
                
                # cv2.line(img12,(center_y,0),(center_y,img12.shape[0]),(0,0,60))
                # plt.imshow(img12,cmap="gray")
                # plt.show()
                # cv2.line(img14,(points[0][1],0),(points[0][1],img14.shape[0]),(0,0,60))
                # for i in range(len(points)-1):
                #     print(points[i][1],points[i][0],points[i+1][1],points[i+1][0])
                #     cv2.line(img14,(points[i][1],points[i][0]),(points[i+1][1],points[i+1][0]),(0,0,60))
                # cv2.line(img14,(center[1],center[0]),(center[1],img14.shape[0]),(0,0,60))

                line_images = []
                img15=img14.copy()
                
                plt.imshow(img14)
                plt.show()
                
               
                
                              
                #edges = cv2.Canny(img12,200,300)
                #plt.imshow(edges,cmap="gray")
                #plt.show()s
                img16 = np.ones((img15.shape[0],img15.shape[1],3), np.uint8)
                img17 = np.ones((img15.shape[0],img15.shape[1],3), np.uint8)
                red= False
                list2=[]
                for i in range(img15.shape[0]):
                    for j in range(img15.shape[1]):
                        if img15[i,j,0]==255 and img15[i,j,1]==0 and img15[i,j,2]==0:
                            red=True
                            img16[i,j,0]=255
                            img16[i,j,1]=255
                            img16[i,j,2]=255

                        if(red==False):
                            print(i,j)
                            img16[i,j,0]=img15[i,j,0]
                            img16[i,j,1]=img15[i,j,1]
                            img16[i,j,2]=img15[i,j,2]
                        else:
                            img16[i,j,0]=255
                            img16[i,j,1]=255
                            img16[i,j,2]=255


                        
                        if(red==True):
                            img17[i,j,0]=img15[i,j,0]
                            img17[i,j,1]=img15[i,j,1]
                            img17[i,j,2]=img15[i,j,2]
                            if img17[i,j,0]==255 and img17[i,j,1]==0 and img17[i,j,2]==0:
                            
                                img17[i,j,0]=255
                                img17[i,j,1]=255
                                img17[i,j,2]=255
                        else:
                            img17[i,j,0]=255
                            img17[i,j,1]=255
                            img17[i,j,2]=255

                        if(j==img15.shape[1]-1):
                            red=False
                plt.imshow(img16)
                plt.show()
                plt.imshow(img17)
                plt.show()
                list2.append(img16)
                list2.append(img17)
                for l in list2:

                    black = np.zeros((l.shape[0],1),np.uint8)
                    black = cv2.bitwise_not(black)
                    #l = cv2.rotate(l, cv2.ROTATE_90_COUNTERCLOCKWISE)
    #blur = cv2.bilateralFilter(gray,9,75,75)
                    gray = cv2.cvtColor(l, cv2.COLOR_BGR2GRAY)
                    #blur = cv2.medianBlur(gray, 3)
                    ret,th4 = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    #th3 = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
                    img_erosion = cv2.erode(th4, kernel, iterations=1) 
                    res = cv2.matchTemplate(black,img_erosion,cv2.TM_SQDIFF_NORMED)
                    for j in range(res.shape[1]):
                        res[0][j] = round(res[0][j],2)
                    lst=[]
                    for j in range(len(res[0])):
                        if res[0][j]!=0:
                            lst.append(j)
                    img = img_erosion[:,min(lst):max(lst)]
                    #img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
                    final_image_list.append(img)
        print(len(final_image_list))
        for i in final_image_list:
            plt.imshow(i,cmap='gray')
            plt.show()                    

                
               
                
               
              
                    #for i in path:

              
                        
                
                #kernel = np.ones((3,3), np.uint8) 
                #black = cv2.bitwise_not(i)
                #erosion_image = cv2.erode(i, kernel, iterations=1)
                #plt.imshow(erosion_image,cmap="gray")
                #plt.show()
                #closing = cv2.morphologyEx(erosion_image, cv2.MORPH_CLOSE, kernel)
                #plt.imshow(closing,cmap="gray")
                #plt.show()
                
              
               
                
    # show the image    
                        
                #black = cv2.bitwise_not(i)
         
                #blur = cv2.GaussianBlur(black,(5,5),0)
        # adaptive threshold
                #th5 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
        # erosion
                #kernel = np.ones((3,3), np.uint8) 
                #img_erosion = cv2.erode(th5, kernel, iterations=1)
                
                
            """ edges = cv2.Canny(closing,200,300)
                plt.imshow(edges,cmap="gray")
                plt.show()
                
                cnts =  cv2.findContours(edges, cv2.RETR_TREE,
                cv2.CHAIN_APPROX_SIMPLE)
                cnts = imutils.grab_contours(cnts)
                print(len(cnts))
                for c in cnts:
    # compute the center of the contour

                    M = cv2.moments(c)
                    if(M["m00"]!=0):
                        cX = int(M["m10"] / M["m00"])
                        cY = int(M["m01"] / M["m00"])
    # draw the contour and center of the shape on the image
            
                        
                    
    # show the erosion_image
                        print(cX,cY)
                        

                        cv2.drawContours(edges, [c], -1, (255,255,255), 2)
                        cv2.circle(edges, (cX, cY),7, (255,255,255), )

                                

                               
                                   
                plt.imshow(edges,cmap='rainbow')
                plt.show()"""
            """ SUM=[]
                SUM1=[]
                for i in peak:
                    SUM.append(xsum_x[0][i])
                    SUM1.append(int(xsum_x[0][i]/2))
                print(SUM)
               
                print(SUM1)"""
                
            """ pixel=[]
                for j  in peak:
                    pi=[]
                    for p in range(ximg2.shape[0]):
                        if(ximg2[p,j].all()==0):

                            pi.append(p)
                    x=int(statistics.median(pi))
                    pixel.append([x,j])        
                print(pixel)
                height= th5.shape[0]
                height=int(height/2)
                
                for j in pixel:
                    
                    cv2.circle(ximg2,(j[1],j[0]),5,[255,255,255],2)
                plt.imshow(ximg2,cmap="gray")
                plt.show()"""        

                

   
            """ for k in range(len(sum_x[0])):
                    for j in peak:
                        if sum_x[0][i].any()==sum_x[0][j].any():
        #count.append()
                            print("found them")

                    #cv2.line(th5,(j,0),(j,th5.shape[0]),(0,0,60))
        #i+=1       
                            cv2.circle(thresh1,(j,height),2,[0,255,0],-1)
                

            
        plt.imshow(thresh1)
        plt.show()"""
       
        
        """points = [] 
        
        print(points)

        vor = Voronoi(points)

        fig = plt.figure(figsize=(20,20))
        ax = fig.add_subplot(111)
        ax.imshow(ndimage.rotate(thresh1, 90))
        voronoi_plot_2d(vor, point_size=10, ax=ax)
        plt.show()"""

        #peaks, _ = find_peaks(sum_x[0], distance=mean/2)
       
        
        
# difference between peaks is >= 150
     
# prints [186 180 177 171 177 169 167 164 158 162 172]
        
       
        
       
      
        
        
        
        

        """for i in img_lst:
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
            plt.show()"""
       

   

   
      
           