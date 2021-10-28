import os
import cv2
input_dir = "/Users/lucky/Desktop/bdd100k/rainy_image/"
output_dir = "/Users/lucky/Desktop/bdd100k/rainy_image_resize/"
file_name_list = os.listdir(input_dir)
for file_name in file_name_list:
    print("Resizing " + file_name)
    img = cv2.imread(input_dir + file_name)
    img = cv2.resize(img, (720, 480), interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(output_dir + file_name, img)