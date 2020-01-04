from enum import IntEnum
from math import sqrt


class BoundingBox:

    def __init__(self, x0, x1, y0, y1):
        self.x0 = x0
        self.x1 = x1
        self.y0 = y0
        self.y1 = y1

    def __str__(self):
        return f'x0:{self.x0} y0:{self.y0} x1:{self.x1} y1:{self.y1}'

    def size(self):
        return abs(self.x1 - self.x0) * abs(self.y1 - self.y0)


class Color(IntEnum):
    GREEN = 0
    YELLOW = 1
    WHITE = 2


class Descriptor:

    def __init__(self, color: Color, box, invariants):
        self.color = color
        self.box = box
        self.invariants = invariants

    def distance(self, model: dict):
        invariants = model[self.color]
        d = 0.0
        for i in range(invariants.shape[1]):
            mean, std = invariants.T[i]
            d += pow(mean - self.invariants[i], 2) / pow(std, 2)
        return sqrt(d)
