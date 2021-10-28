import json
import os 
import time 
import cv2
import math

LABEL_MAP = {
    "car": 0,
    "bus": 0,
    "person": 1,
    "truck": 0,
    "traffic sign": 2,
    "traffic light": 3,
}

intput_img_dir = ['/Users/lucky/Desktop/bdd100k/bdd100k_daytime/train/',
                  '/Users/lucky/Desktop/bdd100k/bdd100k_daytime/val/']
output_img_dir = '/Users/lucky/Desktop/bdd100k/bdd100k_daytime/bdd100k_TDRN_train/'
out_ano_dir = "/Users/lucky/Desktop/bdd100k/bdd100k_daytime/bdd100k_annotation_train/"

# intput_img_dir = ['/Users/lucky/Desktop/bdd100k/bdd100k_daytime/test/']
# output_img_dir = '/Users/lucky/Desktop/bdd100k/bdd100k_daytime/bdd100k_TDRN_test/'
# out_ano_dir = "/Users/lucky/Desktop/bdd100k/bdd100k_daytime/bdd100k_annotation_test/"

raw_json = json.load(open("/Users/lucky/Desktop/bdd100k/labels/bdd100k_labels_images_val.json", "r"))

label = {}
for image_label in raw_json:
  det_list = []
  for det in image_label["labels"]:
    if det["category"] in LABEL_MAP:
      if det["category"] == 'truck' or det["category"] == 'bus':
        class_name = "car"
      elif det["category"] == 'traffic sign':
        class_name = "traffic_sign"
      elif det["category"] == 'traffic light':
        class_name = "traffic_light"
      else:
        class_name = det["category"]
      det_list.append((class_name,
                      round(det["box2d"]['x1']),
                      round(det["box2d"]['y1']),
                      round(det["box2d"]['x2']),
                      round(det["box2d"]['y2'])))
  label[image_label["name"]] = det_list

c = 0
for dir in intput_img_dir:
    for file_name in os.listdir(dir):
        # Create img for TDRN
        img = cv2.imread(dir + file_name)
        img = cv2.resize(img, (320, 320), interpolation=cv2.INTER_AREA)
        new_name = format(c, '06d') + ".jpg"
        cv2.imwrite(output_img_dir + new_name, img)
        print("Writing "  + new_name)
        c += 1

        string_prefix = \
          '<annotation>\n' +\
            '\t<folder>VOC2007</folder>\n' +\
            '\t<filename>' + new_name +'</filename>\n' +\
            '\t<source>\n' +\
              '\t\t<database>The VOC2007 Database</database>\n' +\
              '\t\t<annotation>PASCAL VOC2007</annotation>\n' +\
              '\t\t<image>flickr</image>\n' +\
              '\t\t<flickrid>194986987</flickrid>\n' +\
            '\t</source>\n' +\
            '\t<owner>\n' +\
              '\t\t<flickrid>spiderkiller</flickrid>\n' +\
              '\t\t<name>spiderkiller</name>\n' +\
            '\t</owner>\n' +\
            '\t<size>\n' +\
              '\t\t<width>320</width>\n' +\
              '\t\t<height>320</height>\n' +\
              '\t\t<depth>3</depth>\n' +\
            '\t</size>\n' +\
            '\t<segmented>0</segmented>\n'

        # Create Annotated xml file
        with open(out_ano_dir + new_name.split('.')[0] + '.xml', 'w') as f:
          string = string_prefix
          for det in label[file_name]:
            xmin = math.floor(float(det[1])*0.2500)
            ymin = math.floor(float(det[2])*0.4444)
            xmax = math.floor(float(det[3])*0.2500)
            ymax = math.floor(float(det[4])*0.4444)

            # make bb don't exceed boudary
            if xmin == 320:
              xmin = 319
            if ymin == 320:
              ymin = 319
            if xmax == 320:
              xmax = 319
            if ymax == 320:
              ymax = 319

            # Make sure width and height is more then 1
            if xmin >= xmax or ymin >= ymax:
              print("Skip object, becuase its trivial size")
              continue
              
            
            string += \
            '\t<object>\n' +\
              '\t\t<name>' + det[0] + '</name>\n' +\
              '\t\t<pose>Unspecified</pose>\n' +\
              '\t\t<truncated>1</truncated>\n' +\
              '\t\t<difficult>0</difficult>\n' +\
              '\t\t<bndbox>\n' +\
                '\t\t\t<xmin>' + str(xmin) + '</xmin>\n' +\
                '\t\t\t<ymin>' + str(ymin) + '</ymin>\n' +\
                '\t\t\t<xmax>' + str(xmax) + '</xmax>\n' +\
                '\t\t\t<ymax>' + str(ymax) + '</ymax>\n' +\
              '\t\t</bndbox>\n' +\
            '\t</object>\n'
          string += '</annotation>'
          f.write(string)