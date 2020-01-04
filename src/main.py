import cv2

from analysis import image_invariants
from loader import load_images
from loader import load_model
from logger import logger
from model import Descriptor
from segmentation import segment

logger.info('loading images')
true_logos, false_logos = load_images()
model = load_model()
for img, file in true_logos:
    logger.info(f'extracting roi from {file}')
    objects = segment(img)
    recognized_regions = []
    for color in objects.keys():
        for obj, roi in objects[color]:
            if roi.size() < 500:
                continue
            logger.info(f'calculating object invariants, region: {roi}')
            invariants = image_invariants(obj, lambda px: px == 255)
            descriptor = Descriptor(color, roi, invariants[0:4] + invariants[6:8])
            distance = descriptor.distance(model)
            logger.info(f'calculated invariants {descriptor.invariants}')
            logger.info(f'calculated distance {distance}')
            cv2.imshow('roi', obj)
            cv2.waitKey(0)
            if distance < 2.0:
                logger.info('object recognized')
                recognized_regions.append(descriptor)







