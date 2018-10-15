import cv2
import numpy as np 
import imutils 

def main():
    image_size = 256
    img = cv2.imread("test_image.tif")
    img = cv2.resize(img, (image_size, image_size),0,0, cv2.INTER_LINEAR)
    M = cv2.getRotationMatrix2D((256,256),30,1)
    dst = cv2.warpAffine(img,M,(256,256))

    cv2.imwrite("rotated_image.tif", dst)


def rotate():
    #image_size = 256
    img = cv2.imread("test_image.tif")
    #img = cv2.resize(img, (image_size, image_size),0,0, cv2.INTER_LINEAR)
    rotated = imutils.rotate_bound(img, 30)
    rotated = cv2.resize(rotated, (256,256),0,0,cv2.INTER_LINEAR)
    cv2.imwrite("Correct_rotated_image.tif", rotated)


#main()

rotate()
