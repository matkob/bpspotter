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

            v = c_max
            s = 0 if v == 0 else (v - c_min) / v
            if c_min == c_max:
                h = 0
            elif v == r2:
                h = 60 * (g2 - b2) / (v - c_min)
            elif v == g2:
                h = 120 + 60 * (b2 - r2) / (v - c_min)
            else:
                h = 240 + 60 * (r2 - g2) / (v - c_min)
            if h < 0:
                h = h + 360

            h = int(h / 2)
            s = int(s * 255)
            v = int(v * 255)
            copy[row, col] = [h, s, v]
    return copy
