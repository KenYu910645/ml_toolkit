# Conversion of RGB to LAB(L for lightness and a and b for the color opponents green–red and blue–yellow) will do the work. Apply CLAHE to the converted image in LAB format to only Lightness component and convert back the image to RGB. Here is the snippet.
import cv2 
import os 
input_dir = "/Users/lucky/Desktop/bdd100k/bdd100k_daytime/test/"
output_dir = "/Users/lucky/Desktop/bdd100k/bdd100k_daytime/test_clahe/"

# Get image file names
files = os.listdir(input_dir)

# Init CLAHE
clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))

c = 0
for name in files:
    img = cv2.imread(input_dir + name)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lab_planes = cv2.split(lab)
    lab_planes[0] = clahe.apply(lab_planes[0])
    lab = cv2.merge(lab_planes)
    img_clahe = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    # cv2.imshow("asdf", output)
    cv2.imwrite(output_dir + name, img_clahe)
    # cv2.waitKey(0)
    print("Processed image: " + name)
    c += 1
print("Total " + str(c) + " images processed.")