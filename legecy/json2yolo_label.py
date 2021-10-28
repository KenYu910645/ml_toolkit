import pprint
import json
# Convert BDD100K label json file to yolo format txt

# YOLO label format:
# <object_class> <x> <y> <width> <height>
# (x,y) is the center of rectangle.
# All four value are [0,1], representing the ratio to image.shape

LABEL_MAP = {
    "car": 0,
    "bus": 0,
    "person": 1,
#    "bike": 3,
    "truck": 0,
    # "motor": 5,
    # "train": 6,
    # "rider": 7,
    "traffic sign": 2,
    "traffic light": 3,
}

out_dir = "/Users/lucky/Desktop/bdd100k/bdd100k_daytime/val/"
raw_json = json.load(open("/Users/lucky/Desktop/bdd100k/labels/bdd100k_labels_images_val.json", "r"))

label = {}
for image_label in raw_json:
  det_list = []
  for det in image_label["labels"]:
    if det["category"] in LABEL_MAP:
      det_list.append((det["category"],
                       round(det["box2d"]['x1']),
                       round(det["box2d"]['y1']),
                       round(det["box2d"]['x2']),
                       round(det["box2d"]['y2'])))
  label[image_label["name"]] = det_list

for name in label:
  print("Writing " + name.split('.')[0] + '.txt ....')
  with open(out_dir + name.split('.')[0] + '.txt', 'w') as f:
    for det in label[name]:
      # det = ('car', 492, 290, 506, 306)
      # image shape: 1280 x 720
      class_name = LABEL_MAP[det[0]]
      center_x = ((det[1] + det[3])/2.0)/1280.0
      cetner_y = ((det[2] + det[4])/2.0)/720.0
      width =  (det[3] - det[1])/1280.0
      height = (det[4] - det[2])/720.0
      
      string =  str(class_name) + ' ' + str(center_x) + ' ' + str(cetner_y) + ' ' +\
                str(width) + ' ' + str(height) + '\n'
      f.write(string)
