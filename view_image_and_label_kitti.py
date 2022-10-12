# Draw 2D bounding box on image 

import cv2
import time
import os

# TODO 
# images files 
IMG_DIR = "/home/lab530/KenYu/kitti_mixup_1/training/image_2/"
# Anotations files 
ANO_DIR = "/home/lab530/KenYu/kitti_mixup_1/training/label_2/"

# output image directory (Optional)
OUT_DIR = "/home/lab530/KenYu/tmp/"

img_paths = [os.path.join(IMG_DIR, i) for i in os.listdir(IMG_DIR)]
ano_paths = [os.path.join(ANO_DIR, i) for i in os.listdir(ANO_DIR)]

print("Get " + str(len(img_paths)) + " images in " + IMG_DIR)
print("Get " + str(len(ano_paths)) + " labels in " + ANO_DIR)

# Remove data in output folder
os.system('rm -rf ' + OUT_DIR + '*')

for img_path in img_paths:
    name = os.path.split(img_path)[1].split('.')[0]
    # Load image
    img = cv2.imread(img_path)

    # Load anotations file
    try:
        with open(os.path.join(ANO_DIR, name + '.txt'), encoding = 'utf-8') as f:
            dets = f.read().split('\n')
            # print(dets)
    except FileNotFoundError:
        print("Can't found anotation file: " + str(os.path.join(ANO_DIR, name + '.txt')))
        continue
    
    # Draw anotations on image
    for det in dets:
        d = det.split()
        try:
            class_name = d[0]
            xmin = int(float(d[4]))
            ymin = int(float(d[5]))
            xmax = int(float(d[6]))
            ymax = int(float(d[7]))

        except IndexError:
            print("Something wrong with the index of labels in file: " + name + '.txt')
            continue

        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(img, class_name, (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 0), 1, cv2.LINE_AA)
        
    
    # cv2.imshow("view_image_and_label", img)
    # cv2.waitKey(0)
    cv2.imwrite(os.path.join(OUT_DIR, name + '.jpg'), img)
