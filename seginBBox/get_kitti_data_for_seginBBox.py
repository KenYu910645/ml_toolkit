import os 
import glob
import cv2
import json 
import numpy as np
import shutil
import imageio as io

IMAGE_DIR = "/home/lab530/KenYu/kitti/training/image_2/"
LABEL_DIR = "/home/lab530/KenYu/kitti/training/label_2/"
OUTPUT_DIR = "/home/lab530/KenYu/kitti/training/image_bbox/"
TRAIN_SPLIT_TXT = "/home/lab530/KenYu/visualDet3D/visualDet3D/data/kitti/chen_split/train.txt" # should only use obj from training data
OUTPUT_IMG_SIZE = None # 512

print("Clean output directory : " + str(OUTPUT_DIR))
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
os.mkdir(OUTPUT_DIR)

# Get all training images
with open(TRAIN_SPLIT_TXT, 'r') as f:
    lines = f.read().splitlines()

imgs_path = sorted( [f"{IMAGE_DIR}{l}.png" for l in lines] )

print(f"total number of image = {len(imgs_path)}")

class Object:
    def __init__(self, str_line, idx_img = None):
        # str_line should be 'Car 0.00 0 -1.58 587.19 178.91 603.38 191.75 1.26 1.60 3.56 -1.53 1.89 73.44 -1.60'
        self.idx_img = idx_img # this obj belong to which image 
        self.raw_str = str_line
        sl = str_line.split()
        self.category, self.truncated, self.occluded, self.alpha = sl[0], float(sl[1]), int(sl[2]), float(sl[3])
        self.x_min, self.y_min, self.x_max, self.y_max = [int(float(i)) for i in sl[4:8]]
        self.h, self.w, self.l = [float(i) for i in sl[8:11]]
        self.x_3d, self.y_3d, self.z_3d = [float(i) for i in sl[11:14]]
        self.rot_y = float(sl[14])
        self.area = (self.x_max - self.x_min) * (self.y_max - self.y_min)

obj_count = 0
for img_path in imgs_path:
    img = cv2.imread(img_path)
    img_name = img_path.split('/')[-1].split('.')[0]
    
    # Load objects
    with open(f"{LABEL_DIR}{img_name}.txt") as f:
        gt_lines = f.read().splitlines()
        gt_lines = list(gt_lines for gt_lines in gt_lines if gt_lines) # Delete empty lines
        gts_src = [Object(gt) for gt in gt_lines]
    
    for idx_gt, gt in enumerate(gts_src):
        if gt.category == "Car" and gt.truncated < 0.5 and gt.occluded == 0.0 and gt.area > 3000:
            if OUTPUT_IMG_SIZE != None : 
                img_gt = cv2.resize(img[gt.y_min:gt.y_max, gt.x_min:gt.x_max], (OUTPUT_IMG_SIZE, OUTPUT_IMG_SIZE), interpolation=cv2.INTER_AREA)
            else:
                img_gt = img[gt.y_min:gt.y_max, gt.x_min:gt.x_max]
            cv2.imwrite(f"{OUTPUT_DIR}{img_name}_{idx_gt}.png", img_gt)
            obj_count += 1
    
    # print(f"Processed {img_name}/{len(imgs_path)}")

print(f"Save all bbox image to {OUTPUT_DIR}")
print(f"Total {obj_count} objctes processed")