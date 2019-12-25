from segmentation import split_merge
from color import rgb2hsv
from color import range_filter
from transform import dilate
from transform import erode
import numpy as np
import cv2
import loader


def segment(image):
    kernel = np.ones((3, 3), np.uint8)
    hsv = rgb2hsv(image)
    mask_green = dilate(erode(range_filter(hsv, (40, 0, 0), (80, 255, 240)), kernel), kernel)
    mask_yellow = dilate(erode(range_filter(hsv, (20, 0, 0), (40, 255, 240)), kernel), kernel)
    mask_white = dilate(erode(range_filter(hsv, (0, 0, 160), (180, 50, 255)), kernel), kernel)
    green_objects = split_merge(mask_green, image)
    yellow_objects = split_merge(mask_yellow, image)
    white_objects = split_merge(mask_white, image)
    for obj, box in green_objects:
        if obj.shape[0] * obj.shape[1] > 1000:
            cv2.imshow('roi', obj)
            cv2.waitKey(0)
            cv2.destroyWindow('roi')


true_logos, false_logos = loader.load_images()
for i in true_logos[1:]:
    segment(i)
# segment(cv2.imread('data/true/213_true_1.jpg'))
