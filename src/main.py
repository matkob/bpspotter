from color import rgb2hsv
from color import range_filter
import numpy as np
from segmentation import split_merge
import cv2
import loader


def segment(image):
    cv2.imshow('image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    hsv = rgb2hsv(image)
    mask = range_filter(hsv, (40, 0, 0), (80, 255, 240))
    cv2.imshow('mask', mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    kernel = np.ones((5, 5), np.uint8)
    erosion = cv2.erode(mask, kernel, iterations=2)
    dilation = cv2.dilate(erosion, kernel, iterations=1)
    extracted_objects = split_merge(dilation, image)
    for obj, box in extracted_objects:
        if obj.shape[0] * obj.shape[1] > 1000:
            cv2.imshow('roi', obj)
            cv2.waitKey(0)
            cv2.destroyWindow('roi')


true_logos, false_logos = loader.load_images()
for i in true_logos[1:]:
    segment(i)
# segment(cv2.imread('data/true/213_true_1.jpg'))
