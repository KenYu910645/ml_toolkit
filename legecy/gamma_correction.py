import cv2
import numpy as np
import os 
input_dir = "/Users/lucky/Desktop/bdd100k/bdd100k_dark/"
output_dir = "/Users/lucky/Desktop/bdd100k/bdd100k_gamma_correction/"

# gamma correction
def gamma_correction(img, c=1, g=2.2):
    out = img.copy()
    out /= 255.
    out = (1/c * out) ** (1/g)
    out *= 255
    out = out.astype(np.uint8)
    return out

# Get image file names
files = os.listdir(input_dir)

c = 0
for name in files:
    # Read image
    img = cv2.imread(input_dir + name).astype(np.float)

    # Gammma correction
    img_out = gamma_correction(img)
    cv2.imwrite(output_dir + name, img_out)

    print("Processed image: " + name)
    c += 1
print("Total " + str(c) + " images processed.")