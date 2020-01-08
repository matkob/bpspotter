from collections import defaultdict

import cv2

from analysis import image_invariants
from loader import load_images
from loader import load_model
from logger import logger
from model import Descriptor, Color
from segmentation import segment


def spot_bp_logo():
    true_logos, false_logos = load_images()
    model = load_model()
    for img, file in false_logos:
        cv2.imshow('image', img)
        cv2.waitKey(100)
        logger.info(f'extracting roi from {file}')
        objects = segment(img)
        recognized_regions = defaultdict(list)
        for color in objects.keys():
            for obj, roi in objects[color]:
                if roi.size() < 200:
                    continue
                logger.info(f'calculating {color.name} object invariants, region: {roi}')
                invariants = image_invariants(obj, lambda px: px == 255)
                descriptor = Descriptor(color, roi, invariants)
                distance = descriptor.distance(model, best_invariants(color))
                logger.info(f'distance {distance}')
                if distance_accepted(color, distance):
                    color_edges(img, roi, (255, 255, 255), width=1)
                    cv2.imshow('image', img)
                    cv2.waitKey(100)
                    logger.info('object recognized')
                    recognized_regions[color].append(descriptor)
        for green_d in recognized_regions[Color.GREEN]:
            yellow_d = find_object_inside(green_d.box, recognized_regions[Color.YELLOW])
            if yellow_d is None:
                continue
            white_d = find_object_inside(yellow_d.box, recognized_regions[Color.WHITE])
            if white_d is None:
                continue
            logger.info(f'found bp logo in image {file}')
            color_edges(img, green_d.box, (0, 255, 0), width=3)
            color_edges(img, yellow_d.box, (0, 255, 255), width=3)
            color_edges(img, white_d.box, (0, 0, 0), width=3)
            cv2.imshow('image', img)
            cv2.waitKey(100)
        logger.info('image processed')
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def find_object_inside(box, descriptors):
    for d in descriptors:
        if d.box.inside(box):
            return d
    return None


def find_all_inside(box, descriptors):
    return [d for d in descriptors if d.box.inside(box)]


def best_invariants(color):
    if color is Color.GREEN:
        return [0, 1, 2, 6, 7]
    elif color is Color.YELLOW:
        return [0, 1, 6]
    else:
        return [0, 1, 2, 6, 7]


def color_edges(image, box, color, width=2):
    image[box.y0:box.y0 + width, box.x0:box.x1] = color
    image[box.y1 - width:box.y1, box.x0:box.x1] = color
    image[box.y0:box.y1, box.x0:box.x0 + width] = color
    image[box.y0:box.y1, box.x1 - width:box.x1] = color


def distance_accepted(color, dist):
    if color is Color.GREEN:
        return dist < 3.1
    elif color is Color.YELLOW:
        return dist < 5
    else:
        return dist < 4


spot_bp_logo()







