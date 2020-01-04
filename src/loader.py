from os import listdir

import cv2


def load_images():
    bp_logos = []
    fake_logos = []
    true_dir = 'data/true/'
    false_dir = 'data/false/'
    for file in listdir(true_dir)[13:20]:
        img = cv2.imread(true_dir + file)
        img = normalize_size(img, 720, 1280)
        bp_logos.append((img, file))
    for file in listdir(false_dir)[:1]:
        img = cv2.imread(false_dir + file)
        img = normalize_size(img, 720, 1280)
        fake_logos.append((img, file))
    return bp_logos, fake_logos


def normalize_size(img, max_height, max_width):
    (height, width, channels) = img.shape
    scale = min(max_height / height, max_width / width)
    img = cv2.resize(img, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_AREA)
    return img
