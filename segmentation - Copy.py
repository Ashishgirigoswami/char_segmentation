import os
import sys
import cv2 
import numpy as np
import matplotlib.pyplot as plt
import utilities.hor_functions as hor_functions


function = hor_functions.utility()

img = function.read_img(sys.argv[1])
img = function.preprocessing(img)
img= function.stroke_width_tranform(img)





