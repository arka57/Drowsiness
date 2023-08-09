import torch
from matplotlib import pyplot as plt
import numpy as np
import cv2
import uuid   # Unique identifier
import os
import time

model = torch.hub.load('ultralytics/yolov5', 'custom', path='/home/iter/Documents/Drowsiness/yolov5/runs/train/exp11/weights/last.pt', force_reload=True)
'''img = os.path.join('data', 'images', 'awake.0202b99e-34f2-11ee-bffc-e7c227936be7.jpg')
results = model(img)
results.print()

#%matplotlib inline 
plt.imshow(np.squeeze(results.render()))
plt.show()
'''
cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    
    # Make detections 
    results = model(frame)
    
    cv2.imshow('YOLO', np.squeeze(results.render()))
    
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
