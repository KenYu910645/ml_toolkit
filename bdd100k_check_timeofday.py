# Check time of day by brightness

# Intput 
src_dir = "../bdd100k_new/val/daytime/"
# Output
wrong_list = "./wrong_list.txt"

import os
import cv2
import pprint

d = {}
fn_list = os.listdir(src_dir)
for i, fn in enumerate(fn_list):
    img = cv2.imread(src_dir + fn)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    m = cv2.mean(img)[0]
    d[m] = fn
    print(str(i) + "/" + str(len(fn_list)))

sort_img = [(k,d[k]) for k in sorted(d.keys())]

print("Welcome to bdd100k_check_timeofday.py.")
print("Press 'q' to exit.")
print("Press '>' for next image, and '<' for previous image.")
print("Press 'r' to record wrong image filename.")

w_str = ""
for m, fn in sort_img:
    img = cv2.imread(src_dir + fn)
    print("mean = " + str(m))
    cv2.imshow("check time of day", img)
    
    # Key Event
    key = cv2.waitKey(0) # msec
    if key == ord('q') or key == 27: # Esc or 'q'
        with open(wrong_list, 'w') as f:
            f.write(w_str)
        break
    elif key == 44: # '<'
        i -= 1
    elif key == ord('r'):
        print("Record " + fn)
        w_str += fn + ' '
    else:  # key == 46: # '->'
        i += 1