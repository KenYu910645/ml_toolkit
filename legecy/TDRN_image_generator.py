import os 
import time 

# intput_dir = ['/Users/lucky/Desktop/bdd100k/bdd100k_daytime/train/',
#               '/Users/lucky/Desktop/bdd100k/bdd100k_daytime/val/']
# output_dir = '/Users/lucky/Desktop/bdd100k/bdd100k_daytime/bdd100k_TDRN_train/'
intput_dir = ['/Users/lucky/Desktop/bdd100k/bdd100k_daytime/test/']
output_dir = '/Users/lucky/Desktop/bdd100k/bdd100k_daytime/bdd100k_TDRN_test/'


import cv2
c = 0
for dir in intput_dir:
    for file_name in os.listdir(dir):
        img = cv2.imread(dir + file_name)
        img = cv2.resize(img, (320, 320), interpolation=cv2.INTER_AREA)
        new_name = format(c, '06d') + ".jpg"
        cv2.imwrite(output_dir + new_name, img)
        print("Writing "  + new_name)
        c += 1