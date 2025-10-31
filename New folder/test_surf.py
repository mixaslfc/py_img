import cv2
surf = cv2.xfeatures2d.SURF_create(hessianThreshold=400)
print("ok", surf)