# Conversion of RGB to LAB(L for lightness and a and b for the color opponents green–red and blue–yellow) will do the work. Apply CLAHE to the converted image in LAB format to only Lightness component and convert back the image to RGB. Here is the snippet.
import cv2 
import os 
from matplotlib import pyplot as plt
import numpy as np
from math import exp, sqrt
input_dir = "/Users/lucky/Desktop/bdd100k/filter_result/"
output_dir = "/Users/lucky/Desktop/bdd100k/filter_result/"

def distance(point1,point2):
    return sqrt((point1[0]-point2[0])**2 + (point1[1]-point2[1])**2)

def gaussianLP(D0,imgShape):
    base = np.zeros(imgShape[:2])
    rows, cols = imgShape[:2]
    center = (rows/2,cols/2)
    for x in range(cols):
        for y in range(rows):
            base[y,x] = exp(((-distance((y,x),center)**2)/(2*(D0**2))))
    return base

def gaussianHP(D0,imgShape):
    base = np.zeros(imgShape[:2])
    rows, cols = imgShape[:2]
    center = (rows/2,cols/2)
    for x in range(cols):
        for y in range(rows):
            base[y,x] = 1 - exp(((-distance((y,x),center)**2)/(2*(D0**2))))
    return base

# Get image file names
files = os.listdir(input_dir)

img = cv2.imread(input_dir + "input_image.jpg")
x = 9
kernel = np.ones((x,x),np.float32)/(x**2)
ans_list = [img]
ans_list.append(cv2.filter2D(img,-1,kernel))
ans_list.append(cv2.GaussianBlur(img,(x,x),0))
ans_list.append(cv2.medianBlur(img,x))
ans_list.append(cv2.bilateralFilter(img,x,75,75))
# ans_list.append(cv2.bilateralFilter(img,9,150,150))

cv2.imwrite(output_dir + 'mean_filter_' + str(x) +'.jpg', ans_list[1])
cv2.imwrite(output_dir + 'gaussian_filter_' + str(x) +'.jpg', ans_list[2])
cv2.imwrite(output_dir + 'median_filter_' + str(x) +'.jpg', ans_list[3])
cv2.imwrite(output_dir + 'bilateral_filter_' + str(x) +'.jpg', ans_list[4])

img_output = np.concatenate(ans_list, axis=1) # axis = 0 is vertical
cv2.imshow('img_output', img_output)
cv2.waitKey(0)

# Init CLAHE
# clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))
'''
c = 0
for name in files:
    img = cv2.imread(input_dir + name)
    
    kernel = np.ones((5,5),np.float32)/25
    dst = cv.filter2D(img,-1,kernel)
    plt.subplot(121),plt.imshow(img),plt.title('Original')
    plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(dst),plt.title('Averaging')
    plt.xticks([]), plt.yticks([])
    plt.show()

    # cv2.imshow("asdf", output)
    cv2.imwrite(output_dir + name, img_clahe)
    # cv2.waitKey(0)
    print("Processed image: " + name)
    c += 1
print("Total " + str(c) + " images processed.")
'''