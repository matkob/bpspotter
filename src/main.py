from os import listdir

import cv2
import numpy as np

from segmentation import split_merge


def load_images():
    bp_logos = []
    fake_logos = []
    true_dir = 'data/true/'
    false_dir = 'data/false/'
    for file in listdir(true_dir)[:10]:
        bp_logos.append(cv2.imread(true_dir + file))
    for file in listdir(false_dir)[:1]:
        fake_logos.append(cv2.imread(false_dir + file))
    return bp_logos, fake_logos


def segment(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    hsv = cv2.blur(hsv, (5, 5))
    mask = cv2.inRange(hsv, (28, 15, 0), (70, 255, 255))
    kernel = np.ones((5, 5), np.uint8)
    erosion = cv2.erode(mask, kernel, iterations=2)
    dilation = cv2.dilate(erosion, kernel, iterations=1)
    extracted_objects = split_merge(dilation, image)
    for obj, box in extracted_objects:
        if obj.shape[0] * obj.shape[1] > 1000:
            cv2.imshow('roi', obj)
            cv2.waitKey(0)


true_logos, false_logos = load_images()
for i in true_logos[1:]:
    segment(i)
# segment(cv2.imread('data/true/213_true_1.jpg'))
