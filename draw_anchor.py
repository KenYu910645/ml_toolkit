import cv2
img = cv2.imread("1.jpg")
img = cv2.resize(img, (412,412), interpolation=cv2.INTER_AREA)
cx = img.shape[1]/2
cy = img.shape[0]/2
anchor = [12, 16, 19, 36, 40, 28, 36, 75, 76, 55, 72, 146, 142, 110, 192, 243, 459, 401] # (width, height)
# anchor = [5, 9, 11, 14, 7, 26, 20, 26, 14, 56, 37, 45, 57, 83, 96, 130, 140, 220]
color = (0,0,255)
for i in range(9):
    if i == 3:
        color = (0,255,0)
    elif i == 6: 
        color = (255,0,0)
    cv2.rectangle(img, (int(cx - anchor[i*2]/2), int(cy - anchor[i*2+1]/2)),
                       (int(cx + anchor[i*2]/2), int(cy + anchor[i*2+1]/2)), color, 1)

cv2.imwrite("1_anchor.jpg", img)