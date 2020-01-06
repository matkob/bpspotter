from collections import defaultdict

import cv2

from analysis import image_invariants
from loader import load_images
from loader import load_model
from logger import logger
from model import Descriptor, Color
from segmentation import segment


def spot_bp_logo():
    logger.info('loading images')
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
                if roi.size() < 500:
                    continue
                logger.info(f'calculating {color.name} object invariants, region: {roi}')
                invariants = image_invariants(obj, lambda px: px == 255)
                descriptor = Descriptor(color, roi, invariants)
                distance = descriptor.distance(model, [0, 1, 6])
                logger.info(f'invariants {descriptor.invariants}')
                logger.info(f'distance {distance}')
                cv2.imshow('roi', obj)
                cv2.waitKey(0)
                if distance < 2.0:
                    logger.info('object recognized')
                    recognized_regions[color].append(descriptor)
        for green_d in recognized_regions[Color.GREEN]:
            yellow_objects = find_all_inside(green_d.box, recognized_regions[Color.YELLOW])
            if len(yellow_objects) == 0:
                continue
            white_object = find_object_inside([obj.box for obj in yellow_objects], recognized_regions[Color.WHITE])
            if white_object is not None:
                logger.info(f'found bp logo in image {file}')
            else:
                logger.info(f'bp logo not found in image {file}')


def find_object_inside(boxes, descriptors):
    for box in boxes:
        for d in descriptors:
            if d.box.inside(box):
                return d
    return None


def find_all_inside(box, descriptors):
    return [d for d in descriptors if d.box.inside(box)]


spot_bp_logo()







