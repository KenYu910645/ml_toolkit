''' bdd100k dataset
from bdd100k_origianl_images/
to
<bdd100k_all>
    |
    |---- train
    |---- val
    |---- test
'''
import os.path as osp
import os
import random
from collections import namedtuple
from shutil import copyfile, rmtree

############
### TODO ###
############
# Input
img_dir = "../bdd100k/bdd100k_origianl_images/100k/" # image directory
ano_dir = "../bdd100k/labels/all_label_yolo_format/"
# Output
out_dir = "../bdd100k_all/"

# remove everything that's in output directory
rmtree(out_dir + "train/", ignore_errors=True)
rmtree(out_dir + "val/",   ignore_errors=True)
rmtree(out_dir + "test/",  ignore_errors=True)
print("Deleted directories train/, val/ and test/")

# Create directory stucture
for i in ['train/', 'val/', 'test/']:
    os.mkdir(out_dir + i)
    print("Create directory : " + out_dir + i)

# Load iamge inputs
train_img_paths = [osp.join(img_dir + 'train/', i) for i in os.listdir(img_dir + 'train/')]
valid_img_paths = [osp.join(img_dir + 'val/'  , i) for i in os.listdir(img_dir + 'val')]

print("Total {a} images at {b}.".format(a = len(train_img_paths), b = img_dir + 'train/'))
print("Total {a} images at {b}.".format(a = len(valid_img_paths), b = img_dir + 'val/'))

# Shuffle images order
random.shuffle(train_img_paths)
random.shuffle(valid_img_paths)

# From train/ to test/
print("Coping " + str(len(train_img_paths[:5000])) + " images to " + out_dir + "test/")
[copyfile(i, osp.join(out_dir + "test/", osp.split(i)[1])) for i in train_img_paths[:5000]]
[copyfile(osp.join(ano_dir, osp.split(i)[1].split('.')[0] + '.txt'),
          osp.join(out_dir + "test/", osp.split(i)[1].split('.')[0] + '.txt' )) for i in train_img_paths[:5000]]
# From train/ to train/
print("Coping " + str(len(train_img_paths[5000:])) + " images to " + out_dir + "train/")
[copyfile(i, osp.join(out_dir + "train/", osp.split(i)[1])) for i in train_img_paths[5000:]]
[copyfile(osp.join(ano_dir, osp.split(i)[1].split('.')[0] + '.txt'),
          osp.join(out_dir + "train/", osp.split(i)[1].split('.')[0] + '.txt' )) for i in train_img_paths[5000:]]
# From val/ to test/
print("Coping " + str(len(valid_img_paths[:3000])) + " images to " + out_dir + "test/")
[copyfile(i, osp.join(out_dir + "test/", osp.split(i)[1])) for i in valid_img_paths[:3000]]
[copyfile(osp.join(ano_dir, osp.split(i)[1].split('.')[0] + '.txt'),
          osp.join(out_dir + "test/", osp.split(i)[1].split('.')[0] + '.txt' )) for i in valid_img_paths[:3000]]
# From val/ to val/
print("Coping " + str(len(valid_img_paths[3000:])) + " images to " + out_dir + "val/")
[copyfile(i, osp.join(out_dir + "val/", osp.split(i)[1])) for i in valid_img_paths[3000:]]
[copyfile(osp.join(ano_dir, osp.split(i)[1].split('.')[0] + '.txt'),
          osp.join(out_dir + "val/", osp.split(i)[1].split('.')[0] + '.txt' )) for i in valid_img_paths[3000:]]


