import numpy as np
import os
import matplotlib
matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path
from matplotlib.gridspec import GridSpec
from PIL import Image
from shutil import rmtree

IS_LABELED = True
# KITTI's categories
OBJ_COLOR = {'Cyclist': 'yellow', 
             'Pedestrian': 'cyan', 
             'Van': 'red', 
             'Misc': 'purple', 
             'Truck': 'orange',
             'Car'  : 'red'} # 'green'
VEHICLES = ['Car']

# Nuscene categories
# OBJ_COLOR = {'bicycle': 'yellow', 
#              'pedestrian': 'cyan', 
#              'bus': 'red', 
#              'barrier': 'purple', 
#              'truck': 'orange',
#              'car'  : 'green',
#              'construction_vehicle' : 'blue',
#              'motorcycle' : 'pink',
#              'traffic_cone' : 'red',
#              'trailer' : 'red'}
# # VEHICLES = ['barrier', 'bicycle', 'bus', 'car', 'construction_vehicle', 'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck']
# VEHICLES = ['car']

# # GAC
LABEL_DIR  = "/home/lab530/KenYu/kitti/training/label_2/"
IMAGE_DIR  = "/home/lab530/KenYu/kitti/training/image_2/"
CALIB_DIR  = "/home/lab530/KenYu/kitti/training/calib/"
OUTPUT_DIR = "/home/lab530/KenYu/ml_toolkit/3d_object_detection_visualization/viz_result/compare_new_new/"
# PRED_DIRS = [("GAC", "/home/lab530/KenYu/visualDet3D/exp_output/best/Mono3D/output/validation/data")]

# LABEL_DIR  = "/home/lab530/KenYu/kitti/training/label_2/"
# IMAGE_DIR  = "/home/lab530/KenYu/kitti/training/image_2/"
# CALIB_DIR  = "/home/lab530/KenYu/kitti/training/calib/"
# OUTPUT_DIR = "/home/lab530/KenYu/ml_toolkit/3d_object_detection_visualization/viz_result/GAC_original_no_filter/"
# PRED_DIRS = [("GAC", "/home/lab530/KenYu/visualDet3D/exp_output/baseline_gac_original/Mono3D/output/validation/data/")]

# # KITTI_mixup one file prediction
# LABEL_DIR  = "/home/lab530/KenYu/kitti_seg_1/training/label_2/"
# IMAGE_DIR  = "/home/lab530/KenYu/kitti_seg_1/training/image_2/"
# CALIB_DIR  = "/home/lab530/KenYu/kitti_seg_1/training/calib/"
# OUTPUT_DIR = "/home/lab530/KenYu/ml_toolkit/3d_object_detection_visualization/viz_result/kitti_toy/"
# PRED_DIRS = [("GAC", "exp_output/mixup/kitti_mixup_1/Mono3D/output/test/data/")]

# KITTI_mixup
# LABEL_DIR  = "/home/lab530/KenYu/kitti_seg_1/training/label_2/"
# IMAGE_DIR  = "/home/lab530/KenYu/kitti_seg_1/training/image_2/"
# CALIB_DIR  = "/home/lab530/KenYu/kitti_seg_1/training/calib/"
# OUTPUT_DIR = "/home/lab530/KenYu/ml_toolkit/3d_object_detection_visualization/viz_result/kitti_seg_1/"
# PRED_DIRS = [("GAC", "/home/lab530/KenYu/kitti_seg_1/training/label_2/")]

# Nuscene 
# LABEL_DIR  = "/home/lab530/KenYu/nusc_kitti/training/label_2/"
# IMAGE_DIR  = "/home/lab530/KenYu/nusc_kitti/training/image_2/"
# CALIB_DIR  = "/home/lab530/KenYu/nusc_kitti/training/calib/"
# OUTPUT_DIR = "/home/lab530/KenYu/ml_toolkit/3d_object_detection_visualization/viz_result/nuscene_kitti/"
# PRED_DIRS = [("GAC", "/home/lab530/KenYu/visualDet3D/exp_output/nuscene_kitti/Mono3D/output/validation/data/")]

# KITTI Comparasion
# LABEL_DIR  = "/home/lab530/KenYu/visualDet3D/kitti/training/label_2/"
# IMAGE_DIR  = "/home/lab530/KenYu/visualDet3D/kitti/training/image_2/"
# CALIB_DIR  = "/home/lab530/KenYu/visualDet3D/kitti/training/calib/"
# OUTPUT_DIR = "/home/lab530/KenYu/ml_toolkit/3d_object_detection_visualization/viz_result/nuscene_kitti/"
# PRED_DIRS = [("SMOKE", "/home/lab530/KenYu/SMOKE/tools/logs/inference/kitti_train/data/")]

PRED_DIRS = [("SMOKE"        , "/home/lab530/KenYu/SMOKE/tools/logs/inference/kitti_train/data/"),
             ("MonoGRNet"    , "/home/lab530/KenYu/MonoGRNet/outputs/kittiBox/val_out/val_result/"),
             ("Pseudo-LiDAR" , "/home/lab530/KenYu/ml_toolkit/3d_object_detection_visualization/pseudo_lidar_prediction/"),
             ("MonoFlex"     , "/home/lab530/KenYu/ml_toolkit/3d_object_detection_visualization/monoflex_prediction/"),
             ("DD3D"         , "/home/lab530/KenYu/dd3d/outputs/2cyqwjvr-20220811_163826/inference/final-tta/kitti_3d_val/bbox3d_predictions_standard_format/"),
             ("Ground-aware" , "/home/lab530/KenYu/visualDet3D/exp_output/baseline_gac_original/Mono3D/output/validation/data/"),
             ("Ours"         , "/home/lab530/KenYu/visualDet3D/exp_output/best/Mono3D/output/validation/data/"),]

# PRED_DIRS = [("Faster-RCNN", "/home/lab530/KenYu/mmdetection/faster_rcnn_exps/output/"),
#              ("YOLOv3"     , "/home/lab530/KenYu/mmdetection/yolov3_exps/output/"),
#              ("FCOS"       , "/home/lab530/KenYu/mmdetection/fcos_exps/output/"),
#              ("RetinaNet"  , "/home/lab530/KenYu/mmdetection/retinanet_exps/output/"),
#              ("Ours"       , "/home/lab530/KenYu/visualDet3D/exp_output/baseline_gac_original/Mono3D/output/validation/data/"),
#              ("Ours+DAS"   , "/home/lab530/KenYu/visualDet3D/exp_output/das/Mono3D/output/validation/data/"),]



class detectionInfo(object):
    def __init__(self, line):
        self.name = line[0]

        self.truncation = float(line[1])
        self.occlusion = int(line[2])

        # local orientation = alpha + pi/2
        self.alpha = float(line[3])

        # in pixel coordinate
        self.xmin = float(line[4])
        self.ymin = float(line[5])
        self.xmax = float(line[6])
        self.ymax = float(line[7])

        # height, weigh, length in object coordinate, meter
        self.h = float(line[8])
        self.w = float(line[9])
        self.l = float(line[10])

        # x, y, z in camera coordinate, meter
        self.tx = float(line[11])
        self.ty = float(line[12])
        self.tz = float(line[13])

        # global orientation [-pi, pi]
        self.rot_global = float(line[14])

def compute_birdviewbox(line, shape, scale):
    npline = [np.float64(line[i]) for i in range(1, len(line))]
    h = npline[7] * scale
    w = npline[8] * scale
    l = npline[9] * scale
    x = npline[10] * scale
    y = npline[11] * scale
    z = npline[12] * scale
    rot_y = npline[13]

    R = np.array([[-np.cos(rot_y), np.sin(rot_y)],
                  [ np.sin(rot_y), np.cos(rot_y)]])
    t = np.array([x, z]).reshape(1, 2).T

    x_corners = [0, l, l, 0]  # -l/2
    z_corners = [w, w, 0, 0]  # -w/2

    # This is a bug !!!
    # x_corners += -w / 2
    # z_corners += -l / 2
    x_corners += -l / 2
    z_corners += -w / 2

    # bounding box in object coordinate
    corners_2D = np.array([x_corners, z_corners])
    # rotate
    corners_2D = R.dot(corners_2D)
    # translation
    corners_2D = t - corners_2D
    # in camera coordinate
    corners_2D[0] += int(shape/2)
    corners_2D = (corners_2D).astype(np.int16)
    corners_2D = corners_2D.T

    return np.vstack((corners_2D, corners_2D[0,:]))

def draw_birdeyes(ax2, line, color, title, is_print_conf = False, shape=900):
    shape = shape # 900
    scale = 20
    # Draw GT
    gt_corners_2d = compute_birdviewbox(line, shape, scale)
    
    codes = [Path.LINETO] * gt_corners_2d.shape[0]
    # print(Path.LINETO)
    codes[0] = Path.MOVETO
    codes[-1] = Path.CLOSEPOLY
    pth = Path(gt_corners_2d, codes)
    p = patches.PathPatch(pth, fill=False, color=color, label=title)
    ax2.add_patch(p)
    # Draw conf text
    # if len(line) == 16: # Prediction 
    if is_print_conf:
        conf = round(float(line[-1]), 2)
        ax2.text(max(gt_corners_2d[:, 0]), max(gt_corners_2d[:, 1]),
                str(conf), fontsize=8, color = (1, 0, 0))

def compute_3Dbox(P2, line):
    obj = detectionInfo(line)
    # Draw 2D Bounding Box
    xmin = int(obj.xmin)
    xmax = int(obj.xmax)
    ymin = int(obj.ymin)
    ymax = int(obj.ymax)
    # width = xmax - xmin
    # height = ymax - ymin
    # box_2d = patches.Rectangle((xmin, ymin), width, height, fill=False, color='red', linewidth='3')
    # ax.add_patch(box_2d)

    # Draw 3D Bounding Box

    R = np.array([[ np.cos(obj.rot_global), 0, np.sin(obj.rot_global)],
                  [ 0,                      1, 0                     ],
                  [-np.sin(obj.rot_global), 0, np.cos(obj.rot_global)]])

    x_corners = [0, obj.l, obj.l, obj.l, obj.l, 0,     0,     0]  # -l/2
    y_corners = [0, 0,     obj.h, obj.h, 0,     0,     obj.h, obj.h]  # -h
    z_corners = [0, 0,     0,     obj.w, obj.w, obj.w, obj.w, 0]  # -w/2

    x_corners = [i - obj.l / 2 for i in x_corners]
    y_corners = [i - obj.h     for i in y_corners]
    z_corners = [i - obj.w / 2 for i in z_corners]

    corners_3D = np.array([x_corners, y_corners, z_corners])
    corners_3D = R.dot(corners_3D)
    corners_3D += np.array([obj.tx, obj.ty, obj.tz]).reshape((3, 1))

    # Avoid z_3d < 0, saturate at 0.0001
    corners_3D[2] = np.array([max(0.0001, i) for i in corners_3D[2]])
    # 
    corners_3D_1 = np.vstack((corners_3D, np.ones((corners_3D.shape[-1]))))
    corners_2D = P2.dot(corners_3D_1)
    corners_2D = corners_2D / corners_2D[2]
    corners_2D = corners_2D[:2]
    
    return corners_2D

def draw_3Dbox(ax, P2, line, color, is_print_conf = False):

    corners_2D = compute_3Dbox(P2, line)
    # draw all lines through path
    # https://matplotlib.org/users/path_tutorial.html
    bb3d_lines_verts_idx = [0, 1, 2, 3, 4, 5, 6, 7, 0, 5, 4, 1, 2, 7, 6, 3]
    bb3d_on_2d_lines_verts = corners_2D[:, bb3d_lines_verts_idx]
    verts = bb3d_on_2d_lines_verts.T
    codes = [Path.LINETO] * verts.shape[0]
    codes[0] = Path.MOVETO
    # codes[-1] = Path.CLOSEPOLYq
    pth = Path(verts, codes)
    p = patches.PathPatch(pth, fill=False, color=color, linewidth=1)

    width = corners_2D[:, 3][0] - corners_2D[:, 1][0]
    height = corners_2D[:, 2][1] - corners_2D[:, 1][1]
    
    # put a mask on the front
    front_fill = patches.Rectangle((corners_2D[:, 1]), width, height, fill=True, color=color, alpha=0.4)
    ax.add_patch(p)
    ax.add_patch(front_fill)

    # Print confidence score on prediction bounding box
    if is_print_conf:
        conf = round(float(line[-1]), 2)
        ax.text(max(0, min(corners_2D[0])), 
                max(0, min(corners_2D[1])),
                str(conf), fontsize=10, color = (1, 0, 0))

def draw_2Dbox(ax, line, color, is_print_conf = False):
    x1, y1, x2, y2 = (int(float(line[4])), int(float(line[5])), int(float(line[6])), int(float(line[7])))
    width  = x2 - x1
    height = y2 - y1
    front_fill = patches.Rectangle((x1, y1),
                                    width, 
                                    height, 
                                    fill=False, 
                                    color=color, 
                                    linewidth=1,
                                    alpha=1)
    # Print confidence score on prediction bounding box
    if is_print_conf:
        conf = round(float(line[-1]), 2)
        ax.text(max(0, x1), 
                max(0, y1),
                str(conf), fontsize=10, color = (1, 0, 0))
    ax.add_patch(front_fill)

def draw_3Dgt(ax, P2, line, color):

    corners_2D = compute_3Dbox(P2, line)

    # draw all lines through path
    # https://matplotlib.org/users/path_tutorial.html
    bb3d_lines_verts_idx = [0, 1, 2, 3, 4, 5, 6, 7, 0, 5, 4, 1, 2, 7, 6, 3]
    bb3d_on_2d_lines_verts = corners_2D[:, bb3d_lines_verts_idx]
    verts = bb3d_on_2d_lines_verts.T
    codes = [Path.LINETO] * verts.shape[0]
    codes[0] = Path.MOVETO
    # codes[-1] = Path.CLOSEPOLYq
    pth = Path(verts, codes)
    p = patches.PathPatch(pth, fill=False, color=color, linewidth=2)

    width = corners_2D[:, 3][0] - corners_2D[:, 1][0]
    height = corners_2D[:, 2][1] - corners_2D[:, 1][1]
    # put a mask on the front
    front_fill = patches.Rectangle((corners_2D[:, 1]), width, height, fill=True, color=color, alpha=0.4)
    ax.add_patch(p)
    ax.add_patch(front_fill)

# Clean output directory 
if os.path.exists(OUTPUT_DIR):
    rmtree(OUTPUT_DIR)
os.mkdir(OUTPUT_DIR)

# Find image idx by exploring first method prediction files
dataset = [name.split('.')[0] for name in sorted(os.listdir(PRED_DIRS[0][1]))]

for index in range(len(dataset)):
    # TODO
    # if dataset[index] != "000039": continue
    
    # Create fig
    fig = plt.figure(figsize=(21, 5*(2+len(PRED_DIRS))), dpi=100)
    fig.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0)
    # 
    gs = GridSpec(2+len(PRED_DIRS), 4)
    gs.update(wspace=0)  # set the spacing between axes.
    ax_img = [fig.add_subplot(gs[i, :3]) for i in range(2 + len(PRED_DIRS))]
    ax_bev = [fig.add_subplot(gs[i+1,3]) for i in range(len(PRED_DIRS)+1)]
    # Load image 
    image_file = os.path.join(IMAGE_DIR, dataset[index] + '.png')
    image = Image.open(image_file).convert('RGB')
    
    # Load label file
    label_file = os.path.join(LABEL_DIR, dataset[index] + '.txt')

    # Load calibration file
    calibration_file = os.path.join(CALIB_DIR, dataset[index] + '.txt')
    for line in open(calibration_file):
        if 'P2' in line:
            P2 = line.split(' ')
            P2 = np.asarray([float(i) for i in P2[1:]])
            P2 = np.reshape(P2, (3, 4))

    # Draw Ground true image
    with open(label_file) as f1:
        for line_gt in f1:
            line_gt = line_gt.strip().split(' ')
            # Draw GT 3D bounding box
            # This should only be use when you want to show GAC's original accept gts
            # if line_gt[0] in VEHICLES and int(line_gt[2]) < 2 and float(line_gt[13]) > 3 :
            if line_gt[0] in VEHICLES:
                
                color = OBJ_COLOR[line_gt[0]]
                # draw_2Dbox(ax_img[1], line_gt, color, is_print_conf = False)
                draw_3Dbox(ax_img[1], P2, line_gt, color, is_print_conf = False)
                [draw_birdeyes(a, line_gt, 'orange', 'ground truth', is_print_conf = False, shape = image.size[1]) for a in ax_bev]

    # Draw Prediction bounding box
    for method_idx, method_name in enumerate(PRED_DIRS) :
        prediction_file  = os.path.join(PRED_DIRS[method_idx][1], dataset[index] + '.txt')
        with open(prediction_file) as f2:
            for line_p in f2:
                line_p = line_p.strip().split(' ')
                if line_p[0] in VEHICLES and float(line_p[-1]) > 0.5: # Only print out conf?0.5
                    color = OBJ_COLOR[line_p[0]]
                    # draw_2Dbox(ax_img[method_idx+2], line_p, color,  is_print_conf = True)
                    draw_3Dbox(ax_img[method_idx+2], P2, line_p, color)
                    draw_birdeyes(ax_bev[method_idx+1], line_p, 'green', 'prediction', is_print_conf = True, shape = image.size[1])

    # Draw method_name on canvas    
    for i, m_name in enumerate(['Input', 'Ground Truth'] + list(map(lambda x: x[0], PRED_DIRS))):
        ax_img[i].text(1 , 1, m_name, fontsize=20, color = (1, 1, 0),
                bbox=dict(facecolor='black', boxstyle='round'),
                horizontalalignment='left',
                verticalalignment='top')
    
    # visualize 3D bounding box
    for i in ax_img:
        i.imshow(image)
        i.set_xticks([]) # remove axis value
        i.set_yticks([])

    # Visualize BEV
    shape = image.size[1] # 375 # 9003
    birdimage = np.zeros((shape, shape, 3), np.uint8)
    
    # plot camera view range
    x1 = np.linspace(0, shape / 2)
    x2 = np.linspace(shape / 2, shape)
    for i in ax_bev:
        i.plot(x1, shape / 2 - x1, ls='--', color='grey', linewidth=3, alpha=0.5)
        i.plot(x2, x2 - shape / 2, ls='--', color='grey', linewidth=3, alpha=0.5)
        i.scatter(shape / 2, 0, color="red", s=200 , marker="o")
        
        i.imshow(birdimage, origin='lower')
        i.set_xticks([])
        i.set_yticks([])

    fig.savefig(os.path.join(OUTPUT_DIR, dataset[index]), dpi=fig.dpi, bbox_inches='tight', pad_inches=0)
    print(f"Save figure to {os.path.join(OUTPUT_DIR, dataset[index])}.png")
    plt.close(fig) # avoid comsumer too many memory
