# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 11:18:55 2021

@author: Eva
"""

import numpy as np
import argparse
import cv2


arg = argparse.ArgumentParser()
arg.add_argument("-i", "--image", required=True,
                 help="path to input image")
arg.add_argument("-p", "--prototxt", required=True,
                 help="path to Caffe 'deploy' prototxt file")
arg.add_argument("-m", "--model", required=True,
                 help="path to Caffe pre-trained model")
arg.add_argument("-c", "--confidence", type=float, default=0.5,
                 help="minimum probability to filter weak detections")
args = vars(arg.parse_args())


print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

image = cv2.imread(args["image"])
(h, w) = image.shape[:2]
blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,
                             (300, 300), (104.0, 177.0, 123.0))


print("[INFO] computing object detections...")
net.setInput(blob)

#detection shape: [1, 1, N, 7]
detections = net.forward()

#looping over every detected objects
for i in range(detections.shape[2]):
    #[batchId, class label, confidence, left, top, right, bottom]
    confidence = detections[0, 0, i, 2]
    
    if confidence > args["confidence"]:
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")
        
        text = "{:.2f}%".format(confidence * 100)
        y = startY - 10 if startY - 10 > 10 else startY + 10
        cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 2)
        cv2.putText(image, text,(startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

#img_name = args["image"].split('.')
#cv2.imwrite(img_name[0]+'_detect.'+img_name[1], image)    
cv2.imshow("Output", image)
cv2.waitKey(0)        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
