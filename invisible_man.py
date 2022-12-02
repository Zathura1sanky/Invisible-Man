import cvzone
import cv2
from cvzone.SelfiSegmentationModule import SelfiSegmentation
import numpy as np

back = cv2.imread('real.jpg')
# back = cv2.flip(back,1)
back = cv2.resize(back,(640,480),interpolation=cv2.INTER_CUBIC)
cap = cv2.VideoCapture(0)
cap.set(3,640)
cap.set(4,480)

segmentor = SelfiSegmentation()
imgblack = np.zeros((720,1280,3),dtype='uint8')

while True:
    success,img = cap.read()
    img = cv2.flip(img, 1)
    cv2.imshow('real',img)
    img_out = segmentor.removeBG(img,(0,0,0),threshold=0.1)
    gray = cv2.cvtColor(img_out,cv2.COLOR_BGR2GRAY)
    _,thresh = cv2.threshold(gray,200,255,cv2.THRESH_BINARY_INV)
    thresh = cv2.cvtColor(thresh,cv2.COLOR_GRAY2BGR)
    cv2.imshow('thresh',thresh)
    # stacked   = cvzone.stackImages([img,img_out],2,1)
    # cv2.imshow('stacked',stacked)
    img = cv2.bitwise_and(thresh,back)
    # img = cv2.bitwise_or(back,thresh)

    cv2.imshow('final',img)
    cv2.waitKey(1)