# This script convert kitti 2015 instance segmentation challeage data into seginBBox format

import os
import glob
import cv2
import json
import numpy as np
import shutil
import imageio as io
import scipy.misc as sp

IMAGE_DIR  = "/home/lab530/KenYu/kitti_segmentation/training/image_2/"
LABEL_DIR  = "/home/lab530/KenYu/kitti_segmentation/training/instance/"
OUTPUT_DIR = "/home/lab530/KenYu/seginBBox_kitti2015/"
OUTPUT_IMG_SIZE = 512

print("Clean output directory : " + str(OUTPUT_DIR))
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
os.mkdir(OUTPUT_DIR)
os.mkdir(OUTPUT_DIR + "img/")
os.mkdir(OUTPUT_DIR + "masks/")

# Get all images
imgs_path = sorted(glob.glob(f"{IMAGE_DIR}*.png"))

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



idx_obj = 0
for img_path in imgs_path:
    img = cv2.imread(img_path)
    # print(img)
    img_name = img_path.split('/')[-1].split('.')[0]
    
    # Load objects
    
    # instance_semantic_gt = io.imread(LABEL_DIR + img_name + ".png")
    instance_semantic_gt = cv2.imread(LABEL_DIR + img_name + ".png", cv2.IMREAD_UNCHANGED)
    
    # instance_gt = instance_semantic_gt  % 256
    # semantic_gt = instance_semantic_gt // 256
    
    # print("================ instance_gt =================")
    # hist = cv2.calcHist([instance_gt], [0], None, [256], [0, 256])
    # for value, count in zip(list(range(256)), hist):
    #     if count != 0: 
    #         print(f"{value} value count {count}")

    # print("================ semantic_gt =================")
    # hist = cv2.calcHist([semantic_gt], [0], None, [256], [0, 256])
    # for value, count in zip(list(range(256)), hist):
    #     if count != 0: 
    #         print(f"{value} value count {count}")


    instclassid_list = np.unique(instance_semantic_gt)
    # print(instclassid_list)
    for instclassid in instclassid_list:
        instid = instclassid % 256 
        if instid > 0 :
            classid = instclassid // 256
            if classid == 26: # Instance is a car
                mask = instance_semantic_gt == instclassid
                instance_size = np.count_nonzero(mask)*1.0
                if instance_size < 1500 : continue
                
                # print(f"instance_size = {(classid, instance_size)}")
                mask = mask.astype(np.uint8)*255
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                if len(contours) > 1 : continue

                x, y, w, h = cv2.boundingRect(contours[0])
                # Output img cropped
                # print(img[int(x):int(x+w), int(y):int(y+h)])
                img_rgb_cropped = cv2.resize(img[int(y):int(y+h), int(x):int(x+w)], 
                                             (OUTPUT_IMG_SIZE, OUTPUT_IMG_SIZE), 
                                             interpolation=cv2.INTER_AREA)
                img_mask_cropped = cv2.resize(mask[int(y):int(y+h), int(x):int(x+w)], 
                                             (OUTPUT_IMG_SIZE, OUTPUT_IMG_SIZE), 
                                              interpolation=cv2.INTER_AREA)
                # 
                cv2.imwrite(f"/home/lab530/KenYu/seginBBox_kitti2015/img/{idx_obj}.png", img_rgb_cropped)
                cv2.imwrite(f"/home/lab530/KenYu/seginBBox_kitti2015/masks/{idx_obj}_mask.png", img_mask_cropped)
                idx_obj += 1
    
    print(f"Processed {img_name}/{len(imgs_path)}")