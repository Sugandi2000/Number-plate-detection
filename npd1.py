import cv2
from matplotlib import pyplot as plt
import numpy as np
import imutils
import easyocr
import os

img = cv2.imread('image6.jpeg')
gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

bfilter = cv2.bilateralFilter(gray, 10,20,20) #noice reduction
edged = cv2.Canny(bfilter,30,200) #edge detection

keypoints = cv2.findContours(edged.copy(),cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours = imutils.grab_contours(keypoints)
contours = sorted(contours, key=cv2.contourArea,reverse=True)[:10]

location = None
for contour in contours:
    approx = cv2.approxPolyDP(contour,10,True)
    if len(approx)== 4:
        location = approx
        break
if location is not None:
    mask = np.zeros(gray.shape,np.uint8)
    new_img = cv2.drawContours(mask,[location],0,255,-1)
    new_img = cv2.bitwise_and(img,img,mask=mask)
else:
    print("Number plate not found.")

reader = easyocr.Reader(['en'], gpu= False)
result = reader.readtext(new_img)

top_left = tuple(result[0][0][0])
bottom_right = tuple(result[0][0][2])
text = result [0][1]
font = cv2.FONT_HERSHEY_SIMPLEX

for detection in result:
    top_left = tuple([int(val) for val in detection[0][0]])
    bottom_right = tuple([int(val) for val in detection[0][2]])
    text = detection[1]
    font = cv2.FONT_HERSHEY_SIMPLEX
    print(text)