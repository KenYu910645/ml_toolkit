
inspect_dir = "/Users/lucky/Desktop/bdd100k/bdd100k_daytime/bdd100k_annotation_train/"
# inspect_dir = "/Users/lucky/Desktop/VOCdevkit/VOC2007/Annotations/"
import xml.etree.ElementTree as ET
import os 
for xml in os.listdir(inspect_dir):
    tree = ET.parse(inspect_dir + xml)
    root = tree.getroot()
    
    for obj in root.findall('object'):
        class_name = obj.find('name').text
        bb = obj.find('bndbox')
        x_min = int(bb[0].text)
        y_min = int(bb[1].text)
        x_max = int(bb[2].text)
        y_max = int(bb[3].text)
        if x_min >= x_max:
            print("Inspecting " + xml)
            print("Trivial X cooridnate problem")
        if y_min >= y_max:
            print("Inspecting " + xml)
            print("Trivial Y cooridnate problem")

        if  x_min < 0 or x_min > 319 or\
            y_min < 0 or y_min > 319 or\
            x_max < 0 or x_max > 319 or\
            y_max < 0 or y_max > 319:
            print("Inspecting " + xml)
            print("Exceed legit boudnary")

        # for i in range(4):
        #     if int(bb[i].text) == 319:
        #         print("GGGGGGGGGGGGGGGGGGG")
        # print(int(bb[0].text))
        # bb[1].text
        # bb[2].text
        # bb[3].text
        # result_dic[img_num].append((class_name, "annotate", bb[0].text, bb[1].text, bb[2].text, bb[3].text))
