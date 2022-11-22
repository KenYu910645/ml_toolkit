import cv2
import numpy as np
import os 
import glob
input_dir = "../bdd100k_daytime_train/val/"
output_dir = "../bdd100k_daytime_train/val_darkaug_gamma/"

# gamma correction
def gamma_correction(img, c=1.0, g=0.3): # g = 2.2
    out = img.copy()
    out /= 255.
    out = (1/c * out) ** (1/g)
    out *= 255
    out = out.astype(np.uint8)
    return out

# Get image file names
files = glob.glob(input_dir + "*.jpg")
# files = os.listdir(input_dir)

c = 0
for name in files:
    # Read image
    img = cv2.imread(name).astype(np.float)

    fn = os.path.split(name)[1]
    # Gammma correction
    img_out = gamma_correction(img)
    cv2.imwrite(output_dir + fn, img_out)

    print("Processed image: " + fn)
    c += 1

print("Total " + str(c) + " images processed.")
