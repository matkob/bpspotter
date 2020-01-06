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
    for img, file in true_logos:
        cv2.imshow('image', img)
        cv2.waitKey(0)
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
                if distance < 4.5:
                    logger.info('object recognized')
                    recognized_regions[color].append(descriptor)
        for green_d in recognized_regions[Color.GREEN]:
            yellow_objects = find_all_inside(green_d.box, recognized_regions[Color.YELLOW])
            if len(yellow_objects) == 0:
                continue
            white_object = find_object_inside([obj.box for obj in yellow_objects], recognized_regions[Color.WHITE])
            if white_object is not None:
                logger.info(f'found bp logo in image {file}')
                cv2.imshow('logo', img[green_d.box.y0:green_d.box.y1, green_d.box.x0:green_d.box.x1])
            else:
                logger.info(f'bp logo not found in image {file}')
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def find_object_inside(boxes, descriptors):
    for box in boxes:
        for d in descriptors:
            if d.box.inside(box):
                return d
    return None


def find_all_inside(box, descriptors):
    return [d for d in descriptors if d.box.inside(box)]


def best_invariants(color):
    if color is Color.GREEN:
        return [0, 1, 6]
    elif color is Color.YELLOW:
        return [0]
    else:
        return [0, 6]


spot_bp_logo()







