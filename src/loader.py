from os import listdir
import cv2


def load_images():
    bp_logos = []
    fake_logos = []
    true_dir = 'data/true/'
    false_dir = 'data/false/'
    for file in listdir(true_dir)[:10]:
        img = cv2.imread(true_dir + file)
        img = scale_down(img, 720, 1280)
        bp_logos.append(img)
    for file in listdir(false_dir)[:1]:
        img = cv2.imread(false_dir + file)
        img = scale_down(img, 720, 1280)
        fake_logos.append(img)
    return bp_logos, fake_logos


def scale_down(img, max_height, max_width):
    (height, width, channels) = img.shape
    if height * width > max_height * max_width:
        scale = max_height / height
        img = cv2.resize(img, (int(height * scale), int(width * scale)), interpolation=cv2.INTER_AREA)
    return img
