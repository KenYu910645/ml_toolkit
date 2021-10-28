import cv2
import numpy as np
import os
from shutil import copyfile
input_file_path = ["/Users/lucky/Desktop/bdd100k/bdd100k_dark/",
                   "/Users/lucky/Desktop/bdd100k/bdd100k_clahe/",
                   "/Users/lucky/Desktop/bdd100k/bdd100k_lime/",
                   "/Users/lucky/Desktop/bdd100k/bdd100k_retinex/",
                   "/Users/lucky/Desktop/bdd100k/bdd100k_gamma_correction/",
                   "/Users/lucky/Desktop/bdd100k/bdd100k_enlightenGAN/"]
file_name = "b249e7f2-d619bd69.jpg"
output_path = "/Users/lucky/Desktop/bdd100k/b249e7f2-d619bd69/"

for path in input_file_path:
    title = path.split('_')[1].split('/')[0]
    copyfile(path + file_name, output_path + file_name.split(".")[0] + "_" + title + ".jpg")