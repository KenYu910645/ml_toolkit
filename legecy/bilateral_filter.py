# Conversion of RGB to LAB(L for lightness and a and b for the color opponents green–red and blue–yellow) will do the work. Apply CLAHE to the converted image in LAB format to only Lightness component and convert back the image to RGB. Here is the snippet.
import cv2 
import os 
from matplotlib import pyplot as plt
import numpy as np
from math import exp, sqrt
input_dir = "/Users/lucky/Desktop/bdd100k/bdd100k_enlightenGAN/"
output_dir = "/Users/lucky/Desktop/bdd100k/bdd100k_bilateral/"

# Get image file names
files = os.listdir(input_dir)

c = 0
for name in files:
    img = cv2.imread(input_dir + name)
    cv2.imwrite(output_dir + name, cv2.bilateralFilter(img,9,75,75))
    print("Processed image: " + name)
    c += 1
print("Total " + str(c) + " images processed.")