import pprint # for prettier print
import json
from shutil import copyfile 

# pick images from bdd100k original images

# YOLO label format:
# <object_class> <x> <y> <width> <height>
# (x,y) is the center of rectangle.
# All four value are [0,1], representing the ratio to image.shape

# LABEL_MAP = {
#     "car": 0,
#     "bus": 0,
#     "person": 1,
# #    "bike": 3,
#     "truck": 0,
#     # "motor": 5,
#     # "train": 6,
#     # "rider": 7,
#     "traffic sign": 2,
#     "traffic light": 3,
# }
out_dir = "/Users/lucky/Desktop/bdd100k/bdd100k_dark/"
src_dir = "/Users/lucky/Desktop/bdd100k/bdd100K_origianl_images/100k/"
raw_json = json.load(open("/Users/lucky/Desktop/bdd100k/labels/bdd100k_labels_images_val.json", "r"))
# bdd100k_labels_images_train is too much for windows

c = 0
count_copy = 0
for image_label in raw_json:
    ## print meta infor.
    print("==============" + image_label['name'] + "============") # 'c3c0f47b-ac19bef3.jpg'
    print("weather : " + image_label['attributes']['weather']) # 'clear', 'overcast', 'rainy', 'snowy', 'undefined'
    print("scene : " + image_label['attributes']['scene']) # 'city street', 'highway', 'residential', 'parking lot'
    print("timeofday : " + image_label['attributes']['timeofday']) # 'night', 'daytime', 'dawn/dusk'
    print("Number of detection: " + str(len(image_label["labels"])))
    
    # TODO specify your condition here.
    if image_label['attributes']['timeofday'] == "night":
        c += 1
        try:
            copyfile(src_dir + "test/" + image_label['name'], out_dir + image_label['name'])
            count_copy += 1
            continue
        except FileNotFoundError:
            pass
        
        try:
            copyfile(src_dir + "train/" + image_label['name'], out_dir + image_label['name'])
            count_copy += 1
            continue
        except FileNotFoundError:
            pass
        
        try:
            copyfile(src_dir + "val/" + image_label['name'], out_dir + image_label['name'])
            count_copy += 1
            continue
        except FileNotFoundError:
            print(image_label['name'] + " file not found.")

print("Found " + str(c) + " night images in label.")
print("Copy "  + str(count_copy) + " images")
