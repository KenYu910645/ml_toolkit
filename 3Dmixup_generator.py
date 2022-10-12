import cv2
import os 
import glob
import random 
import shutil
from iou_3d import get_3d_box, box3d_iou, box2d_iou

random.seed(5278)

VEHICLES = ["Car"] # What kind of object added to base image
N_DATASET_REPEAT = 5 # The total size of output image = N_DATASET_REPEAT * 7481
N_ADD_OBJ = 1 # How many newly added object in a image

IMAGE_DIR = "/home/lab530/KenYu/visualDet3D/kitti/training/image_2/"
LABEL_DIR = "/home/lab530/KenYu/visualDet3D/kitti/training/label_2/"
CALIB_DIR = "/home/lab530/KenYu/visualDet3D/kitti/training/calib/"
TRAIN_SPLIT_TXT = "/home/lab530/KenYu/visualDet3D/visualDet3D/data/kitti/chen_split/train.txt"
VALID_SPLIT_TXT = "/home/lab530/KenYu/visualDet3D/visualDet3D/data/kitti/chen_split/val.txt"
OUT_IMAGE_DIR = f"/home/lab530/KenYu/kitti_mixup_{N_ADD_OBJ}/training/image_2/"
OUT_LABEL_DIR = f"/home/lab530/KenYu/kitti_mixup_{N_ADD_OBJ}/training/label_2/"
OUT_CALIB_DIR = f"/home/lab530/KenYu/kitti_mixup_{N_ADD_OBJ}/training/calib/"
OUT_SPLIT_DIR = f"/home/lab530/KenYu/visualDet3D/visualDet3D/data/kitti/kitti_mixup_{N_ADD_OBJ}_split/"

# TODO 2D IOU check might make it a good way to do it ? 
# TODO some crop images contain more than one object, will this be a problem?
# TODO if enable multiple objs spawn in sigle image, we need to check if their collide with each other 

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
        self.corners = get_3d_box((self.l, self.w, self.h),
                                   self.rot_y,
                                  (self.x_3d, self.y_3d, self.z_3d))

# Clean output directory 
print("Clean output directory : " + str(OUT_IMAGE_DIR))
print("Clean output directory : " + str(OUT_LABEL_DIR))
print("Clean output directory : " + str(OUT_CALIB_DIR))
print("Clean output directory : " + str(OUT_SPLIT_DIR))
shutil.rmtree(OUT_IMAGE_DIR, ignore_errors=True)
shutil.rmtree(OUT_LABEL_DIR, ignore_errors=True)
shutil.rmtree(OUT_CALIB_DIR, ignore_errors=True)
shutil.rmtree(OUT_SPLIT_DIR, ignore_errors=True)
os.mkdir(OUT_IMAGE_DIR)
os.mkdir(OUT_LABEL_DIR)
os.mkdir(OUT_CALIB_DIR)
os.mkdir(OUT_SPLIT_DIR)

# Record image that can't augumented 
tartar = []

# Only augumented training split 
img_paths = []
with open(TRAIN_SPLIT_TXT, 'r') as f:
    lines = f.read().splitlines()
    img_idxs = list(lines for lines in lines if lines) # Delete empty lines
img_paths = [IMAGE_DIR + l + '.png' for l in lines]
print(f"Find {len(img_paths)} source images.")

# Copy validation split to destination, don't augument
with open(VALID_SPLIT_TXT, 'r') as f:
    lines = f.read().splitlines()
    lines = list(lines for lines in lines if lines) # Delete empty lines
for l in lines:
    shutil.copyfile(IMAGE_DIR + l + ".png", OUT_IMAGE_DIR + l + ".png")
    shutil.copyfile(LABEL_DIR + l + ".txt", OUT_LABEL_DIR + l + ".txt")
    shutil.copyfile(CALIB_DIR + l + ".txt", OUT_CALIB_DIR + l + ".txt")

# Output split file. To specify how to split train and valid data
with open(OUT_SPLIT_DIR + 'train.txt', 'w') as f:
    for idx_repeat in range(N_DATASET_REPEAT):
        [f.write(f"{idx_repeat}{i[1:]}\n") for i in img_idxs]

shutil.copyfile(VALID_SPLIT_TXT, OUT_SPLIT_DIR + "val.txt")
print(f"Ouptut train.txt to {OUT_SPLIT_DIR}")
print(f"Ouptut val.txt to {OUT_SPLIT_DIR}")

shuffle_list = [i.split('/')[-1].split('.')[0] for i in img_paths]
#
for idx_repeat in range(N_DATASET_REPEAT):
    for idx_src, img_path in zip(img_idxs, img_paths):
        # Skip tartar images
        if idx_src in tartar: continue 
        
        # Load image
        img_src = cv2.imread(img_path)
        
        # Add N_ADD_OBJ objects in the source image
        for obj_idx in range(N_ADD_OBJ):
            # Get source objects
            with open(LABEL_DIR + f"{idx_src}.txt") as f:
                gt_lines = f.read().splitlines()
                gt_lines = list(gt_lines for gt_lines in gt_lines if gt_lines) # Delete empty lines
                gts_src = [Object(gt) for gt in gt_lines]
            
            random.shuffle(shuffle_list)
            for idx_add in shuffle_list:
                # Get add_object
                with open(LABEL_DIR + f"{idx_add}.txt") as f:
                    gt_lines = f.read().splitlines()
                    gt_lines = list(gt_lines for gt_lines in gt_lines if gt_lines) # Delete empty lines
                    gts_add = [Object(gt, idx_add) for gt in gt_lines]
                
                # Filter inappropiate objs
                gts_add_tmp = []
                for gt in gts_add:
                    if  gt.category in VEHICLES and gt.truncated < 0.5 and gt.occluded == 0.0 and\
                        gt.area > 3000:
                        gts_add_tmp.append(gt)
                if len(gts_add_tmp) == 0: continue
                gts_add = gts_add_tmp
                random.shuffle(gts_add)

                # Avoid 3D bbox collide with each other on BEV
                gt_rst = None # Result
                for gt_add in gts_add:
                    is_good_spawn = True

                    # Avoid using objects farer than 25 meter
                    if gt_add.z_3d > 25 : continue

                    for gt_src in gts_src:
                        # Get 2D IOU
                        iou_2d = box2d_iou((gt_add.x_min, gt_add.y_min, gt_add.x_max, gt_add.y_max),
                                           (gt_src.x_min, gt_src.y_min, gt_src.x_max, gt_src.y_max))
                        # Get 3D IOU
                        iou_3db, iou_bev = box3d_iou(gt_src.corners, gt_add.corners)

                        # Avoid far object paste on nearer object
                        if iou_2d > 0.0 and gt_add.z_3d > gt_src.z_3d: is_good_spawn = False

                        # Avoid 3D bbox collide with each other on BEV
                        if iou_bev > 0.0: is_good_spawn = False
                        #
                        if not is_good_spawn: break
                    
                    if is_good_spawn: gt_rst = gt_add
                # 
                if gt_rst != None: break

            # If couldn't find any augmented image, copy paste image without modification
            if gt_rst == None:
                tartar.append(idx_src)
                print(f"[WARNING] Cannot find any suitable gt to add to {idx_src}.png")
                break

            # Get aug image
            img_add = cv2.imread(IMAGE_DIR + f"{gt_rst.idx_img}.png")
            # Check if img_src and img_add has the same resolution
            if img_src.shape != img_add.shape:
                h_src, w_src, _ = img_src.shape
                h_add, w_add, _ = img_add.shape

                # 2D boudning box Saturation. 
                gt_rst.x_max = min(gt_rst.x_max, w_add-1)
                gt_rst.y_max = min(gt_rst.y_max, h_add-1)
                
                #
                x_min_new = int(gt_rst.x_min * (w_src/w_add))
                x_max_new = int(gt_rst.x_max * (w_src/w_add))
                y_min_new = int(gt_rst.y_min * (h_src/h_add))
                y_max_new = int(gt_rst.y_max * (h_src/h_add))
                
                #
                roi_add = cv2.resize(img_add[gt_rst.y_min:gt_rst.y_max, gt_rst.x_min:gt_rst.x_max], 
                                    (x_max_new - x_min_new, y_max_new - y_min_new),
                                    interpolation=cv2.INTER_AREA)
                # Add new object on augmented image and .txt
                img_src[y_min_new:y_max_new, x_min_new:x_max_new] = 0.5*roi_add + 0.5*img_src[y_min_new:y_max_new, x_min_new:x_max_new]
            else:
                # Add new object on augmented image
                img_src[gt_rst.y_min:gt_rst.y_max, gt_rst.x_min:gt_rst.x_max] = 0.5*img_add[gt_rst.y_min:gt_rst.y_max, gt_rst.x_min:gt_rst.x_max] +\
                                                                                0.5*img_src[gt_rst.y_min:gt_rst.y_max, gt_rst.x_min:gt_rst.x_max]
            # Add new object in label.txt
            gts_src.append(gt_rst)
        
        # Output calib.txt, directly copy source
        shutil.copyfile(CALIB_DIR + f"{idx_src}.txt", OUT_CALIB_DIR + f"{idx_repeat}{idx_src[1:]}.txt")

        # Output label.txt, TODO 2 obj, need to check !!!
        with open(OUT_LABEL_DIR + f"{idx_repeat}{idx_src[1:]}.txt", 'w') as f:
            # s = ""
            # for gt in gts_src: s += gt.raw_str + '\n'
            # f.write(s)
            for gt in gts_src: f.write(gt.raw_str + '\n')
        
        # Output image.png
        cv2.imwrite(OUT_IMAGE_DIR + f"{idx_repeat}{idx_src[1:]}.png", img_src)
        # 
        print(f"Processed {idx_src}/{len(img_paths)} image of {idx_repeat+1}/{N_DATASET_REPEAT} repeats")

print(f"tartar = {tartar}")
print(f"There are {len(tartar)} images can't be augumented!")