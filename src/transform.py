import numpy as np


def morph(image, kernel, operator):
    copy = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    (kernel_height, kernel_width) = kernel.shape
    img_height, img_width = image.shape[0], image.shape[1]
    h_margin, w_margin = int((kernel_height - 1) / 2), int((kernel_width - 1) / 2)
    assert kernel_height % 2 == 1 and kernel_width % 2 == 1
    for col in range(w_margin, img_width - w_margin):
        for row in range(h_margin, img_height - h_margin):
            points = image[row - h_margin:row + h_margin + 1, col-w_margin:col+w_margin + 1]
            copy[row, col] = operator(points)
    return copy


def dilate(image, kernel):
    return morph(image, kernel, lambda o: o.max())


def erode(image, kernel):
    return morph(image, kernel, lambda o: o.min())
