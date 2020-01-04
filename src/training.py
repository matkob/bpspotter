import os
from os import listdir

import cv2

import loader
from analysis import image_invariants
from logger import logger
from segmentation import segment

if not os.path.exists('data/train_green'):
    os.makedirs('data/train_green')
if not os.path.exists('data/train_yellow'):
    os.makedirs('data/train_yellow')
if not os.path.exists('data/train_white'):
    os.makedirs('data/train_white')


def save_examples():
    true_logos, false_logos = loader.load_images()
    for i in true_logos:
        name = i[1]
        logger.info(name)
        green, yellow, white = segment(i[0])
        logger.info('presenting green objects of interest')
        id = 0
        for obj, box in green:
            if obj.shape[0] * obj.shape[1] > 1000:
                cv2.imshow('roi', obj)
                if cv2.waitKey(0) == 121:
                    cv2.imwrite(f'data/train_green/{name}_{id}', obj)
                    id += 1
                cv2.destroyWindow('roi')
        logger.info('presenting yellow objects of interest')
        id = 0
        for obj, box in yellow:
            if obj.shape[0] * obj.shape[1] > 1000:
                cv2.imshow('roi', obj)
                if cv2.waitKey(0) == 121:
                    cv2.imwrite(f'data/train_yellow/{name}_{id}', obj)
                    id += 1
                cv2.destroyWindow('roi')
        logger.info('presenting white objects of interest')
        id = 0
        for obj, box in white:
            if obj.shape[0] * obj.shape[1] > 100:
                cv2.imshow('roi', obj)
                if cv2.waitKey(0) == 121:
                    cv2.imwrite(f'data/train_white/{name}_{id}', obj)
                    id += 1
                cv2.destroyWindow('roi')
        cv2.destroyAllWindows()


def analyse_examples():
    green_dir = 'data/train_green/'
    yellow_dir = 'data/train_yellow/'
    white_dir = 'data/train_white/'
    print('green')
    for file in listdir(green_dir):
        img = cv2.imread(green_dir + file, cv2.IMREAD_GRAYSCALE)
        invariants = image_invariants(img, lambda px: px == 255)
        print(file)
        print(invariants)
    print('yellow')
    for file in listdir(yellow_dir):
        img = cv2.imread(yellow_dir + file, cv2.IMREAD_GRAYSCALE)
        invariants = image_invariants(img, lambda px: px == 255)
        print(file)
        print(invariants)
    print('white')
    for file in listdir(white_dir):
        img = cv2.imread(white_dir + file, cv2.IMREAD_GRAYSCALE)
        invariants = image_invariants(img, lambda px: px == 255)
        print(file)
        print(invariants)


analyse_examples()
