# from bdd100k_all/test split into daytime and nighttime

import glob
import os
from shutil import copyfile, rmtree
import json

# TODO 
# Input 
src_dir = "../bdd100k_all/test/"
print("Loading json file")
raw_json = json.load(open("../bdd100k_all/bdd100k_labels_images_train.json", "r")) + json.load(open("../bdd100k_all/bdd100k_labels_images_val.json", "r"))
# raw_json.update(raw_json_val)
# Output
out_daytime_dir = "../bdd100k_daytime_train/"
out_night_dir = "../bdd100k_all/test_only_night/"

# Clean directory
for i in [out_daytime_dir, out_night_dir]:
    print("Cleaning directories : " + i)
    rmtree(i, ignore_errors=True)
    print("Create directory : " + i)
    os.mkdir(i)

# Get test/ filename
fn_dict = set()
for fp in glob.glob(src_dir + "*.jpg"):
    fn = os.path.split(fp)[1]
    fn_dict.add(fn)

c_dict = {'daytime':0, 'night':0, 'dawn/dusk':0, 'undefined':0}
for image_label in raw_json:
    fn = image_label['name']
    if fn in fn_dict: # Is in test/ 
        time_of_day = image_label['attributes']['timeofday']  # 'night': 0, 'daytime': 0, 'dawn/dusk': 0, 'undefined': 0
        c_dict[time_of_day] += 1
        if time_of_day == 'daytime':
            copyfile(src_dir + fn, out_daytime_dir + fn) # Copy image
            copyfile(src_dir + fn[:-4] + '.txt', out_daytime_dir + fn[:-4] + '.txt') # Copy label
        elif time_of_day == 'night':
            copyfile(src_dir + fn, out_night_dir + fn) # Copy image
            copyfile(src_dir + fn[:-4] + '.txt', out_night_dir + fn[:-4] + '.txt') # Copy label
print(c_dict)
