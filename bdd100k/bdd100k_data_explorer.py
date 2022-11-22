import pprint
import json
from collections import defaultdict

LABEL_MAP = {
    "car": 0,
    "bus": 1,
    "person": 2,
#    "bike": 3,
    "truck": 4,
    # "motor": 5,
    # "train": 6,
    # "rider": 7,
    "traffic sign": 8,
    "traffic light": 9,
}
out_dir = "."
print("Loading json file .....")
raw_json = json.load(open("../bdd100k/labels/bdd100k_labels_images_train.json", "r")) # total 69863 labels
# raw_json = json.load(open("../bdd100k/labels/bdd100k_labels_images_val.json", "r")) # total 10000 labels

import os
name_set = set()
for image_label in raw_json:
    name_set.add(image_label['name'])
print("Total number of labels: " + str(len(name_set)))
w_dict = defaultdict(int) # {'clear': 0, 'overcast': 0, 'rainy': 0, 'snowy': 0, 'partly cloudy' : 0, 'undefined': 0}
t_dict = defaultdict(int) # {'night': 0, 'daytime': 0, 'dawn/dusk': 0, 'undefined': 0}
s_dict = defaultdict(int) # {'city street': 0, 'highway': 0, 'residential': 0, 'parking lot': 0, 'undefined': 0}

for image_label in raw_json:
    w_dict[image_label['attributes']['weather']] += 1
    t_dict[image_label['attributes']['timeofday']] += 1
    s_dict[image_label['attributes']['scene']] += 1
pprint.pprint(w_dict)
pprint.pprint(t_dict)
pprint.pprint(s_dict)

c_dict = defaultdict(int)
for image_label in raw_json:
    for det in image_label["labels"]:
        c_dict[det["category"]] += 1
pprint.pprint(c_dict)



# find unmatched image and labels 
# c = 0
# s = ''
# for fn in os.listdir('../bdd100k/bdd100k_origianl_images/100k/train/'):
#     if fn not in name_set:
#         # w_list.append(fn)
#         s += (fn + ' ')
#         c += 1
# print("Total {a} unmatched images in train/".format(a = str(c)))
# print("Try command : ")
# print("$rm -f " + s)