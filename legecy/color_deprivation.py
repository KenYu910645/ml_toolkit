import cv2
import numpy as np
import os 
import glob
input_dir = "../bdd100k_daytime_train/train/"
# input_dir = "../bdd100k_daytime_train/val/"
output_dir = "../bdd100k_daytime_train/train_darkaug_color/"
# output_dir = "../bdd100k_daytime_train/val_darkaug_color/"


# Get image file names
files = glob.glob(input_dir + "*.jpg")

c = 0
for name in files:
    # Read image
    img = cv2.imread(name).astype(np.float)
    fn = os.path.split(name)[1]
    # Gamma correction

    imgR = np.zeros(img.shape)
    imgG = np.zeros(img.shape)
    imgB = np.zeros(img.shape)

    #assign the red channel of src to empty image
    imgR[:,:,2] = img[:,:,2]
    imgG[:,:,1] = img[:,:,1]
    imgB[:,:,0] = img[:,:,0]

    # cv2.imwrite(output_dir + fn, red_img)
    cv2.imwrite(output_dir + fn.split('.')[0] + "_RED." + fn.split('.')[1], imgR)
    cv2.imwrite(output_dir + fn.split('.')[0] + "_GREEN." + fn.split('.')[1], imgG)
    cv2.imwrite(output_dir + fn.split('.')[0] + "_BLUE." + fn.split('.')[1], imgB)

    print("Processed image: " + fn)
    c += 1

print("Total " + str(c) + " images processed.")
