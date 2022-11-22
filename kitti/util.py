import numpy as np 

def kitti_label_file_parser(label_file_path):
    with open(label_file_path) as f:
        lines = f.read().splitlines()
        lines = list(lines for lines in lines if lines) # Delete empty lines
    idx_img = label_file_path.split('/')[-1].split('.')[0]
    return [KITTI_Object(str_line, idx_img, idx_line) for idx_line, str_line in enumerate(lines)]

class KITTI_Object:
    def __init__(self, str_line, idx_img = None, idx_line = None):
        # str_line should be 'Car 0.00 0 -1.58 587.19 178.91 603.38 191.75 1.26 1.60 3.56 -1.53 1.89 73.44 -1.60'
        # idx_img = "000123"
        # idx_line is i-th line in the label.txt
        self.idx_img  = idx_img # this obj belong to which image 
        self.idx_line = idx_line # this obj belong to which line in label.txt
        self.raw_str = str_line
        sl = str_line.split()
        self.category, self.truncated, self.occluded, self.alpha = sl[0], float(sl[1]), int(sl[2]), float(sl[3])
        self.x_min, self.y_min, self.x_max, self.y_max = [int(float(i)) for i in sl[4:8]]
        self.h, self.w, self.l = [float(i) for i in sl[8:11]]
        self.x_3d, self.y_3d, self.z_3d = [float(i) for i in sl[11:14]]
        self.rot_y = float(sl[14])
        self.area = (self.x_max - self.x_min) * (self.y_max - self.y_min)
    
    def __str__(self):
        return self.raw_str

def get_corner_2D(P2, loc_3d, rot_y, dimension):
    # Get corner that project to 2D image plane
    x3d, y3d, z3d = loc_3d
    l, h, w = dimension
    
    R = np.array([[ np.cos(rot_y), 0, np.sin(rot_y)],
                  [ 0,             1, 0            ],
                  [-np.sin(rot_y), 0, np.cos(rot_y)]])

    x_corners = [0, l, l, l, l, 0, 0, 0]  # -l/2
    y_corners = [0, 0, h, h, 0, 0, h, h]  # -h
    z_corners = [0, 0, 0, w, w, w, w, 0]  # -w/2

    x_corners = [i - l / 2 for i in x_corners]
    y_corners = [i - h     for i in y_corners]
    z_corners = [i - w / 2 for i in z_corners]

    corners_3D = np.array([x_corners, y_corners, z_corners])
    corners_3D = R.dot(corners_3D)
    corners_3D += np.array([x3d, y3d, z3d]).reshape((3, 1))

    # Avoid z_3d < 0, saturate at 0.0001
    corners_3D[2] = np.array([max(0.0001, i) for i in corners_3D[2]])
    # 
    corners_3D_1 = np.vstack((corners_3D, np.ones((corners_3D.shape[-1]))))
    corners_2D = P2.dot(corners_3D_1)
    corners_2D = corners_2D / corners_2D[2]
    corners_2D = corners_2D[:2]
    
    return corners_2D

def draw_birdeyes(ax2, line, color, title, is_print_conf = False):
    shape = 900
    scale = 15
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

def kitti_calib_file_parser(calib_file_path):
    with open(calib_file_path) as f:
        lines = f.read().splitlines()
        for line in lines:
            if 'P2:' in line.split():
                P2 = np.array([float(i) for i in line.split()[1:]] )
                P2 = np.reshape(P2, (3,4))
                return P2

