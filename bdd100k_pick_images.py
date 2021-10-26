import pprint # for prettier print
import json
from shutil import copyfile 

# pick images from bdd100k original images with specified condition.

# YOLO label format:
# <object_class> <x> <y> <width> <height>
# (x,y) is the center of rectangle.
# All four value are [0,1], representing the ratio to image.shape

# Intput 
src_dir = "../bdd100k/bdd100k_origianl_images/100k/val/"
raw_json = json.load(open("../bdd100k/labels/bdd100k_labels_images_val.json", "r"))
# Output
out_dir = "../bdd100k_new/val/daytime/"

c = 0
count_copy = 0
for image_label in raw_json:
    ## print meta infor.
    # print("==============" + image_label['name'] + "============") # 'c3c0f47b-ac19bef3.jpg'
    # print("weather : " + image_label['attributes']['weather']) # 'clear', 'overcast', 'rainy', 'snowy', 'undefined'
    # print("scene : " + image_label['attributes']['scene']) # 'city street', 'highway', 'residential', 'parking lot'
    # print("timeofday : " + image_label['attributes']['timeofday']) # 'night', 'daytime', 'dawn/dusk'
    # print("Number of detection: " + str(len(image_label["labels"])))
    
    # TODO specify your condition here.
    if image_label['attributes']['timeofday'] == "daytime":
        c += 1
        try:
            copyfile(src_dir + image_label['name'], out_dir + image_label['name'])
            count_copy += 1
            continue
        except FileNotFoundError:
            print("File not found: " + image_label['name'])

print("Found " + str(c) + " images in label.")
print("Copy "  + str(count_copy) + " images")
