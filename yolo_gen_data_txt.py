# Input 
# src_path = "../bdd100k_all/train/"
src_path = "../bdd100k_all/val/"
# Output
# out_path = "../darknet/data/bdd100k_train.txt"
out_path = "../darknet/data/bdd100k_val.txt"

import os
# image_files = []
s = ''
for fn in os.listdir(src_path):
    if fn.endswith(".jpg"):
        s += os.path.abspath(src_path + fn) + '\n'

with open(out_path, "w") as f:
    f.write(s)
