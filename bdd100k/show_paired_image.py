
import cv2
import numpy as np
import os
import glob

input_file_path = "../bdd100k_daytime_train/train_darkaug_color/"
output_dir = "/mnt/c/Users/spide/Desktop/tmp/"

img_list = glob.glob(input_file_path + "*.jpg")
# print(img_list)
for fn in img_list:
    if fn.find("_RED.jpg") == -1 and fn.find("_GREEN.jpg") == -1 and fn.find("_BLUE.jpg") == -1:
        img1 = cv2.imread(fn)
        img2 = cv2.imread(fn[:-4] + "_RED.jpg")
        img3 = cv2.imread(fn[:-4] + "_GREEN.jpg")
        img4 = cv2.imread(fn[:-4] + "_BLUE.jpg")
        img_com = np.concatenate([img1, img2, img3, img4], axis=1) # axis = 0 is vertical
        cv2.imwrite(output_dir + os.path.split(fn)[1], img_com)
