import pprint
import json
from shutil import rmtree
import os 
# Convert BDD100K label json file to yolo format txt

# YOLO label format:
# <object_class> <x> <y> <width> <height>
# (x,y) is the center of rectangle.
# All four value are [0,1], representing the ratio to image.shape

# Input 
raw_json = json.load(open("../bdd100k/labels/bdd100k_labels_images_train.json", "r"))
# raw_json = json.load(open("../bdd100k/labels/bdd100k_labels_images_val.json", "r"))
# Output
out_dir  = "../bdd100k/labels/train_label_yolo_format/"
# out_dir  = "../bdd100k/labels/valid_label_yolo_format/"

LABEL_MAP = {
    "car": 0,
    "bus": 0,
    "person": 1,
#    "bike": 3,
    "truck": 0,
    # "motor": 5,
    # "train": 6,
    # "rider": 7,
    "traffic sign": 2,
    "traffic light": 3,
}

# Clean up output directory
rmtree(out_dir, ignore_errors=True)
os.mkdir(out_dir)
print("Create directory : " + out_dir)

for image_label in raw_json:
    s = ""
    for det in image_label["labels"]:
        if det["category"] in LABEL_MAP:
            x1 = round(det["box2d"]['x1'])
            y1 = round(det["box2d"]['y1'])
            x2 = round(det["box2d"]['x2'])
            y2 = round(det["box2d"]['y2'])

            class_name = LABEL_MAP[det["category"]]
            center_x = ((x1 + x2)/2.0)/1280.0
            cetner_y = ((y1 + y2)/2.0)/720.0
            width =  (x2 - x1)/1280.0
            height = (y2 - y1)/720.0
            s += str(class_name) + ' ' + str(center_x) + ' ' + str(cetner_y) + ' ' +\
                 str(width) + ' ' + str(height) + '\n'

    # Write string to file.
    with open(out_dir + image_label["name"].split('.')[0] + '.txt', 'w') as f:
        f.write(s)
