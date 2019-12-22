import numpy as np
from matplotlib import pyplot as plt
from os import listdir
import cv2
import statistics


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
    # label_regions(dilation, 100, 255)
    split_merge(dilation, 100, 255)
    result = cv2.bitwise_and(image, image, mask=dilation)
    cv2.imshow('hsv', result)
    cv2.waitKey(0)


def label_regions(image, low, high, step=5):
    assert step != 1
    regions = []
    mask = np.zeros(image.shape, dtype=int)
    height, width = image.shape
    for row in range(height):
        for col in range(width):
            brightness = image[row, col]
            mask[row, col] = 1 if low <= brightness <= high else 0
    b = step
    for row in range(0, height, int(height / 10)):
        for col in range(0, width, int(width / 10)):
            flood_fill(mask, col, row, b)
            b += step
    for color in range(step, b, step):
        x0, y0, x1, y1 = (0, 0, 0, 0)
        for row in range(height):
            for col in range(width):
                if mask[row, col] == color:
                    x0 = x0 if x0 < col else col
                    x1 = x1 if x1 > col else col
                    y0 = y0 if y0 < row else row
                    y1 = y1 if y1 > row else row
        regions.append(image[y0:y1, x0:x1])
    return regions


def is_uniform(image):
    unique = np.unique(image)
    if len(unique) == 1:
        return True, unique[0]
    return False, None


def split_merge(image, low, high, step=5):
    assert step != 1
    height, width = image.shape
    parts = [(image, 0, 0, width, height)]
    regions = []

    while len(parts) > 0:
        part, x0, y0, x1, y1 = parts.pop(0)
        height = y1 - y0
        width = x1 - x0
        uniform, value = is_uniform(part)
        if uniform and value == 255:
            regions.append([(x0, y0, x1, y1)])
        elif not uniform:
            half_height = int(height / 2)
            half_width = int(width / 2)
            if half_width > 0 and half_height > 0:
                parts.append((part[0:half_height, 0:half_width], x0, y0, x0 + half_width, y0 + half_height))
                parts.append((part[0:half_height, half_width:width], x0 + half_width, y0, x1, y0 + half_height))
                parts.append((part[half_height:height, 0:half_width], x0, y0 + half_height, x0 + half_width, y1))
                parts.append((part[half_height:height, half_width:width], x0 + half_width, y0 + half_height, x1, y1))
    print(len(regions))
    # regions = np.array(regions, copy=False)
    while True:
        region_group = regions.pop(0)
        neighbours = []
        for index in range(len(regions)):
            other_group = regions[index]
            if other_group is region_group:
                continue
            if is_neighbour(region_group, other_group):
                neighbours.append((other_group, index))

        for neighbour, index in neighbours:
            regions.pop(index)
            region_group += neighbour
        regions.append(region_group)
        

def is_neighbour(region_group1, region_group2):
    for region1 in region_group1:
        for region2 in region_group2:
            if (region1[2] > region2[0] and region1[0] < region2[2] and (region1[1] == region2[3] or region1[3] == region2[1]))\
            or (region1[3] > region2[1] and region1[1] < region2[3] and (region1[0] == region2[2] or region1[2] == region2[0])):
                return True
    return False


def flood_fill(mask, x, y, color):
    height, width = mask.shape
    if x == width or y == height or mask[y, x] != 1:
        return
    mask[y, x] = color
    flood_fill(mask, x + 1, y, color)
    flood_fill(mask, x - 1, y, color)
    flood_fill(mask, x, y + 1, color)
    flood_fill(mask, x, y - 1, color)
    return


true_logos, false_logos = load_images()
for i in true_logos:
    segment(i)
# segment(cv2.imread('data/true/213_true_1.jpg'))
