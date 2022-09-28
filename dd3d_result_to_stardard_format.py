# This script transform dd3d's result(json) to stardard format 
import json
import os 
from collections import defaultdict
from pyquaternion import Quaternion
from shutil import rmtree
import numpy as np

INPUT_JSON = "/home/lab530/KenYu/dd3d/outputs/2cyqwjvr-20220811_163826/inference/final-tta/kitti_3d_val/bbox3d_predictions.json"
OUTPUT_DIR = "/home/lab530/KenYu/dd3d/outputs/2cyqwjvr-20220811_163826/inference/final-tta/kitti_3d_val/bbox3d_predictions_standard_format/"


class Pose:
    """SE(3) rigid transform class that allows compounding of 6-DOF poses
    and provides common transformations that are commonly seen in geometric problems.
    """
    def __init__(self, wxyz=np.float32([1., 0., 0., 0.]), tvec=np.float32([0., 0., 0.])):
        """Initialize a Pose with Quaternion and 3D Position

        Parameters
        ----------
        wxyz: np.float32 or Quaternion (default: np.float32([1,0,0,0]))
            Quaternion/Rotation (wxyz)

        tvec: np.float32 (default: np.float32([0,0,0]))
            Translation (xyz)
        """
        assert isinstance(wxyz, (np.ndarray, Quaternion))
        assert isinstance(tvec, np.ndarray)

        if isinstance(wxyz, np.ndarray):
            assert np.abs(1.0 - np.linalg.norm(wxyz)) < 1.0e-3

        self.quat = Quaternion(wxyz)
        self.tvec = tvec

    def __repr__(self):
        formatter = {'float_kind': lambda x: '%.2f' % x}
        tvec_str = np.array2string(self.tvec, formatter=formatter)
        return 'wxyz: {}, tvec: ({})'.format(self.quat, tvec_str)

    def copy(self):
        """Return a copy of this pose object.

        Returns
        ----------
        result: Pose
            Copied pose object.
        """
        return self.__class__(Quaternion(self.quat), self.tvec.copy())

    def __mul__(self, other):
        """Left-multiply Pose with another Pose or 3D-Points.

        Parameters
        ----------
        other: Pose or np.ndarray
            1. Pose: Identical to oplus operation.
               (i.e. self_pose * other_pose)
            2. ndarray: transform [N x 3] point set
               (i.e. X' = self_pose * X)

        Returns
        ----------
        result: Pose or np.ndarray
            Transformed pose or point cloud
        """
        if isinstance(other, Pose):
            assert isinstance(other, self.__class__)
            t = self.quat.rotate(other.tvec) + self.tvec
            q = self.quat * other.quat
            return self.__class__(q, t)
        elif isinstance(other, np.ndarray):
            assert other.shape[-1] == 3, 'Point cloud is not 3-dimensional'
            X = np.hstack([other, np.ones((len(other), 1))]).T
            return (np.dot(self.matrix, X).T)[:, :3]
        else:
            return NotImplemented

    def __rmul__(self, other):
        raise NotImplementedError('Right multiply not implemented yet!')

    def inverse(self):
        """Returns a new Pose that corresponds to the
        inverse of this one.

        Returns
        ----------
        result: Pose
            Inverted pose
        """
        qinv = self.quat.inverse
        return self.__class__(qinv, qinv.rotate(-self.tvec))

    @property
    def matrix(self):
        """Returns a 4x4 homogeneous matrix of the form [R t; 0 1]

        Returns
        ----------
        result: np.ndarray
            4x4 homogeneous matrix
        """
        result = self.quat.transformation_matrix
        result[:3, 3] = self.tvec
        return result

    @property
    def rotation_matrix(self):
        """Returns the 3x3 rotation matrix (R)

        Returns
        ----------
        result: np.ndarray
            3x3 rotation matrix
        """
        result = self.quat.transformation_matrix
        return result[:3, :3]

    @property
    def rotation(self):
        """Return the rotation component of the pose as a Quaternion object.

        Returns
        ----------
        self.quat: Quaternion
            Rotation component of the Pose object.
        """
        return self.quat

    @property
    def translation(self):
        """Return the translation component of the pose as a np.ndarray.

        Returns
        ----------
        self.tvec: np.ndarray
            Translation component of the Pose object.
        """
        return self.tvec

    @classmethod
    def from_matrix(cls, transformation_matrix):
        """Initialize pose from 4x4 transformation matrix

        Parameters
        ----------
        transformation_matrix: np.ndarray
            4x4 containing rotation/translation

        Returns
        -------
        Pose
        """
        return cls(wxyz=Quaternion(matrix=transformation_matrix[:3, :3]), tvec=np.float32(transformation_matrix[:3, 3]))

    @classmethod
    def from_rotation_translation(cls, rotation_matrix, tvec):
        """Initialize pose from rotation matrix and translation vector.

        Parameters
        ----------
        rotation_matrix : np.ndarray
            3x3 rotation matrix
        tvec : np.ndarray
            length-3 translation vector
        """
        return cls(wxyz=Quaternion(matrix=rotation_matrix), tvec=np.float64(tvec))

    def __eq__(self, other):
        return self.quat == other.quat and (self.tvec == other.tvec).all()

# This is from https://github.com/TRI-ML/dd3d/blob/86d8660c29612b79836dad9b6c39972ac2ca1557/tridet/evaluators/kitti_3d_evaluator.py#L205
def convert_3d_box_to_kitti(box):
    """Convert a single 3D bounding box (GenericBoxes3D) to KITTI convention. i.e. for evaluation. We
    assume the box is in the reference frame of camera_2 (annotations are given in this frame).

    Usage:
        >>> box_camera_2 = pose_02.inverse() * pose_0V * box_velodyne
        >>> kitti_bbox_params = convert_3d_box_to_kitti(box_camera_2)

    Parameters
    ----------
    box: GenericBoxes3D
        Box in camera frame (X-right, Y-down, Z-forward)

    Returns
    -------
    W, L, H, x, y, z, rot_y, alpha: float
        KITTI format bounding box parameters.
    """
    assert len(box) == 10

    quat = Quaternion(box[:4])
    tvec = box[4:7]
    sizes = box[7:10]

    # quat = Quaternion(*box.quat.cpu().tolist()[0])
    # tvec = box.tvec.cpu().numpy()[0]
    # sizes = box.size.cpu().numpy()[0]

    # Re-encode into KITTI box convention
    # Translate y up by half of dimension
    tvec += np.array([0., sizes[2] / 2.0, 0])

    inversion = Quaternion(axis=[1, 0, 0], radians=np.pi / 2).inverse
    quat = inversion * quat

    # Construct final pose in KITTI frame (use negative of angle if about positive z)
    if quat.axis[2] > 0:
        kitti_pose = Pose(wxyz=Quaternion(axis=[0, 1, 0], radians=-quat.angle), tvec=tvec)
        rot_y = -quat.angle
    else:
        kitti_pose = Pose(wxyz=Quaternion(axis=[0, 1, 0], radians=quat.angle), tvec=tvec)
        rot_y = quat.angle

    # Construct unit vector pointing in z direction (i.e. [0, 0, 1] direction)
    # The transform this unit vector by pose of car, and drop y component, thus keeping heading direction in BEV (x-z grid)
    v_ = np.float64([[0, 0, 1], [0, 0, 0]])
    v_ = (kitti_pose * v_)[:, ::2]

    # Getting positive theta angle (we define theta as the positive angle between
    # a ray from the origin through the base of the transformed unit vector and the z-axis
    theta = np.arctan2(abs(v_[1, 0]), abs(v_[1, 1]))

    # Depending on whether the base of the transformed unit vector is in the first or
    # second quadrant we add or subtract `theta` from `rot_y` to get alpha, respectively
    alpha = rot_y + theta if v_[1, 0] < 0 else rot_y - theta
    # Bound from [-pi, pi]
    if alpha > np.pi:
        alpha -= 2.0 * np.pi
    elif alpha < -np.pi:
        alpha += 2.0 * np.pi
    alpha = np.around(alpha, decimals=2)  # KITTI precision

    # W, L, H, x, y, z, rot-y, alpha
    return sizes[0], sizes[1], sizes[2], tvec[0], tvec[1], tvec[2], rot_y, alpha

with open(INPUT_JSON, "r") as f:
    dets = json.load(f)
    print(f"Number of Detection: {len(dets)}")     
    
    output_dict = defaultdict(list)
    for i in range(len(dets)):
        fn = dets[i]['file_name'].split('/')[-1].split('.')[0] # 000001
        category = dets[i]['category'] # "Car"
        score = dets[i]['score_3d']
        box_2d = dets[i]['bbox']
        box_3d = dets[i]['bbox3d']

        # Convert 2D box
        x_l, y_l, w, h = dets[i]['bbox']
        l, t, r, b = (x_l, y_l, x_l+w, y_l+h)

        # Convert 3Dbox
        W, L, H, x, y, z, rot_y, alpha = convert_3d_box_to_kitti(box_3d)
        line = f"{category} -1 -1 {alpha} {l} {t} {r} {b} {H} {W} {L} {x} {y} {z} {rot_y} {score}"
        output_dict[fn].append(line)

print(f"Number of images: {len(output_dict)}")
# print(output_dict)

# Clean output directory 
print("Clean output directory : " + OUTPUT_DIR)
rmtree(OUTPUT_DIR, ignore_errors=True)
os.mkdir(OUTPUT_DIR)

# Output standard format to output directory
for fn in output_dict:
    with open(OUTPUT_DIR + fn + ".txt", "w") as f:
        f.write("\n".join(output_dict[fn]))

print(f"Output standard format to {OUTPUT_DIR}")