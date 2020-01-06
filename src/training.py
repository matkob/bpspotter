import os
from os import listdir

import cv2

import loader
from analysis import image_invariants
from logger import logger
from segmentation import segment


def save_examples():
    true_logos, false_logos = loader.load_images()
    for img, name in true_logos:
        cv2.imshow('img', img)
        cv2.waitKey(0)
        logger.info(name)
        objects = segment(img)
        for color in objects.keys():
            if not os.path.exists(f'data/train_{color.name}'):
                os.makedirs(f'data/train_{color.name}')
        for color in objects.keys():
            i = 0
            logger.info(f'presenting {color.name} objects of interest')
            for obj, box in objects[color]:
                if obj.shape[0] * obj.shape[1] > 400:
                    cv2.imshow('roi', obj)
                    if cv2.waitKey(0) == 121:
                        cv2.imwrite(f'data/train_{color.name}/{name}_{i}', obj)
                        i += 1
                    cv2.destroyWindow('roi')
        cv2.destroyAllWindows()


def create_model(filename):
    green_dir = 'data/train_green/'
    yellow_dir = 'data/train_yellow/'
    white_dir = 'data/train_white/'
    with open(f'data/{filename}.csv', 'w') as f:
        for file in listdir(green_dir):
            img = cv2.imread(green_dir + file, cv2.IMREAD_GRAYSCALE)
            invariants = image_invariants(img, lambda px: px > 200)
            line = '\t'.join(['0'] + [str(i) for i in invariants])
            f.write(f'{line}\n')
        for file in listdir(yellow_dir):
            img = cv2.imread(yellow_dir + file, cv2.IMREAD_GRAYSCALE)
            invariants = image_invariants(img, lambda px: px > 200)
            line = '\t'.join(['1'] + [str(i) for i in invariants])
            f.write(f'{line}\n')
        for file in listdir(white_dir):
            img = cv2.imread(white_dir + file, cv2.IMREAD_GRAYSCALE)
            invariants = image_invariants(img, lambda px: px > 200)
            line = '\t'.join(['2'] + [str(i) for i in invariants])
            f.write(f'{line}\n')


save_examples()
