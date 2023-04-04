import os 
import glob
import cv2
import json 
import numpy as np
import shutil
import imageio as io


IMAGE_DIR = "/home/lab530/KenYu/SOLO/cityscapes/leftImg8bit/"
LABEL_DIR = "/home/lab530/KenYu/SOLO/cityscapes/gtFine_trainvaltest/gtFine/"
OUTPUT_DIR = "/home/lab530/KenYu/seginBBox_cityscape/"
OUTPUT_IMG_SIZE = 512

print("Clean output directory : " + str(OUTPUT_DIR))
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
os.mkdir(OUTPUT_DIR)
os.mkdir(OUTPUT_DIR+'img')
os.mkdir(OUTPUT_DIR+'masks')

# Get all images
imgs_path = []
for i in os.listdir(IMAGE_DIR + 'train/'):
    imgs_path += glob.glob(f"{IMAGE_DIR}train/{i}/*.png")
for i in os.listdir(IMAGE_DIR + 'val/'):
    imgs_path += glob.glob(f"{IMAGE_DIR}val/{i}/*.png")
print(f"total number of image = {len(imgs_path)}")

idx_out = 0
for idx, img_path in enumerate(imgs_path):
    img = cv2.imread(img_path)
    img_name = img_path[img_path.rfind('/')+1:img_path.rfind('_')] # darmstadt_000058_000019

    # Load label mask
    img_color       = cv2.imread(f"{LABEL_DIR}{img_path.split('/')[-3]}/{img_path.split('/')[-2]}/{img_name}_gtFine_color.png")
    img_instanceids = cv2.imread(f"{LABEL_DIR}{img_path.split('/')[-3]}/{img_path.split('/')[-2]}/{img_name}_gtFine_instanceIds.png")
    img_labelids    = cv2.imread(f"{LABEL_DIR}{img_path.split('/')[-3]}/{img_path.split('/')[-2]}/{img_name}_gtFine_labelIds.png")

    # hist = cv2.calcHist([img_instanceids], [0], None, [256], [0, 256])
    # for value, count in zip(list(range(256)), hist):
    #     if count != 0: 
    #         print(f"{value} value count {count}")
    
    with open(f"{LABEL_DIR}{img_path.split('/')[-3]}/{img_path.split('/')[-2]}/{img_name}_gtFine_polygons.json", newline='') as f:
        data = json.load(f)
        h = data['imgHeight']
        w = data['imgWidth']
        # print(f"number of objects = {len(data['objects'])}")
        for idx_gt, gt in enumerate(data['objects']):
            if gt['label'] == 'car':
                xmin = np.array(gt['polygon'])[:,0].min()
                xmax = np.array(gt['polygon'])[:,0].max()
                ymin = np.array(gt['polygon'])[:,1].min()
                ymax = np.array(gt['polygon'])[:,1].max()

                # Saturate coordinate
                xmin = max(0, xmin)
                ymin = max(0, ymin)
                xmax = min(w, xmax)
                ymax = min(h, ymax)

                contours = np.array(gt['polygon'])
                area = (xmax - xmin) * (ymax - ymin)

                if area < 3000 : continue

                img_mask = np.zeros(img.shape[:2]) # , dtype=np.int8) # (1024, 2048)
                cv2.drawContours(img_mask, [contours], -1, 255, -1)
                
                # print((ymin, ymax, xmin, xmax))

                img_gt   = cv2.resize(img[ymin:ymax, xmin:xmax],      (OUTPUT_IMG_SIZE, OUTPUT_IMG_SIZE), interpolation=cv2.INTER_AREA)
                img_mask = cv2.resize(img_mask[ymin:ymax, xmin:xmax], (OUTPUT_IMG_SIZE, OUTPUT_IMG_SIZE))
                img_mask = img_mask.astype(np.uint8)

                cv2.imwrite(f"{OUTPUT_DIR}/img/{idx_out}.png", img_gt)
                # io.mimsave(f"{OUTPUT_DIR}/masks/{idx_out}_mask.gif", img_mask)
                cv2.imwrite(f"{OUTPUT_DIR}/masks/{idx_out}_mask.png", img_mask)
                
                idx_out += 1
    print(f"Processed {idx}/{len(imgs_path)}")