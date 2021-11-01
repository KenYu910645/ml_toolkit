import cv2
import time
import os
from shutil import rmtree

# TODO 
# images files 
img_dir = "../image_2/"
# Anotations files 
# ano_dir = "/home/spiderkiller/squeezedet-keras/training/label_2"
ano_dir = "/home/spiderkiller/kitti_dataset/label_2_yolo_format/"
# output image directory (Optional)
# out_dir = "/mnt/c/Users/spide/Desktop/tmp/"
out_dir = "tmp/"

img_paths = [os.path.join(img_dir, i) for i in os.listdir(img_dir)]
ano_paths = [os.path.join(ano_dir, i) for i in os.listdir(ano_dir)]

print("Get " + str(len(img_paths)) + " images in " + img_dir)
print("Get " + str(len(ano_paths)) + " labels in " + ano_dir)

# Clean output directory 
print("Clean output directory : " + str(out_dir))
rmtree(out_dir, ignore_errors=True)
os.mkdir(out_dir)

for img_path in img_paths:
    name = os.path.split(img_path)[1].split('.')[0]
    # Load image
    img = cv2.imread(img_path)
    h, w, _ = img.shape
    # Load anotations file
    try:
        with open(os.path.join(ano_dir, name + '.txt'), encoding = 'utf-8') as f:
            dets = f.read().split('\n')
            # print(dets)
    except FileNotFoundError:
        print("Can't found anotation file: " + str(os.path.join(ano_dir, name + '.txt')))
        continue
    
    # Draw anotations on image
    for det in dets:
        d = det.split()

        try:
            class_name = d[0]
            center_x = float(d[1])*w
            center_y = float(d[2])*h
            bb_w = float(d[3])*w
            bb_h = float(d[4])*h

        except IndexError:
            # print("Something wrong with the index of labels in file: " + name + '.txt')
            pass
        else:
            xmin = int(center_x - bb_w/2)
            ymin = int(center_y - bb_h/2)
            xmax = int(center_x + bb_w/2)
            ymax = int(center_y + bb_h/2)
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 1, cv2.LINE_AA)
            cv2.putText(img, class_name, (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 0), 1, cv2.LINE_AA)
    
    cv2.imwrite(os.path.join(out_dir, name + '.jpg'), img)
    print("Write " + os.path.join(out_dir, name + '.jpg'))
