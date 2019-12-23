import numpy as np


def rgb2hsv(image):
    copy = np.array(image, copy=True)
    height, width, channels = copy.shape
    assert channels == 3
    for col in range(width):
        for row in range(height):
            b, g, r = copy[row, col]
            r2 = float(r) / 255
            b2 = float(b) / 255
            g2 = float(g) / 255
            c_max = max(r2, g2, b2)
            c_min = min(r2, g2, b2)
            d = float(c_max - c_min)
            if d == 0:
                h = 0
            elif c_max == r2:
                h = 60 * (((g2 - b2) / d) % 6)
            elif c_max == g2:
                h = 60 * (((b2 - r2) / d) + 2)
            else:
                h = 60 * (((r2 - g2) / d) + 4)

            h = int(h / 2)
            s = 0 if c_max == 0 else int(d / c_max * 255)
            v = int(c_max * 255)
            copy[row, col] = [h, s, v]
    return copy
