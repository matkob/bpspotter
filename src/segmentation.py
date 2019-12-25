import numpy as np
from model import BoundingBox


def is_uniform(image):
    unique = np.unique(image)
    if len(unique) == 1:
        return True, unique[0]
    return False, None


def is_neighbour(region_group1, region_group2):
    for region1 in region_group1:
        for region2 in region_group2:
            if (region1[2] > region2[0] and region1[0] < region2[2] and (
                    region1[1] == region2[3] or region1[3] == region2[1])) \
                    or (region1[3] > region2[1] and region1[1] < region2[3] and (
                    region1[0] == region2[2] or region1[2] == region2[0])):
                return True
    return False


def split(parts):
    regions = set()
    while len(parts) > 0:
        part, x0, y0, x1, y1 = parts.pop(0)
        height = y1 - y0
        width = x1 - x0
        uniform, value = is_uniform(part)
        if uniform and value != 0:
            regions.add(frozenset([(x0, y0, x1, y1)]))
        elif not uniform:
            half_height = int(height / 2)
            half_width = int(width / 2)
            if half_width > 0 and half_height > 0:
                parts.append((part[0:half_height, 0:half_width], x0, y0, x0 + half_width, y0 + half_height))
                parts.append((part[0:half_height, half_width:width], x0 + half_width, y0, x1, y0 + half_height))
                parts.append((part[half_height:height, 0:half_width], x0, y0 + half_height, x0 + half_width, y1))
                parts.append((part[half_height:height, half_width:width], x0 + half_width, y0 + half_height, x1, y1))
    return regions


def merge(regions):
    merged_regions = []
    while len(regions) > 0:
        region_group = regions.pop()
        neighbours = []
        for other_group in regions:
            if other_group is region_group:
                continue
            if is_neighbour(region_group, other_group):
                neighbours.append(other_group)

        for neighbour in neighbours:
            regions.remove(neighbour)
        if len(neighbours) == 0:
            merged_regions.append(region_group)
        else:
            regions.add(region_group.union(*neighbours))
    print(f'found {len(merged_regions)} regions')
    return merged_regions


def extract(regions, image):
    roi = []
    for region_group in regions:
        min_x = min([r[0] for r in region_group])
        min_y = min([r[1] for r in region_group])
        max_x = max([r[2] for r in region_group])
        max_y = max([r[3] for r in region_group])
        extracted_region = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        box = BoundingBox(min_x, max_x, min_y, max_y)
        for region in region_group:
            extracted_region[region[1]:region[3], region[0]:region[2]] = 255  # image[region[1]:region[3], region[0]:region[2]]
        roi.append((extracted_region[min_y:max_y, min_x:max_x], box))
    return roi


def split_merge(mask, image):
    y1, x1 = mask.shape
    parts = [(mask, 0, 0, x1, y1)]
    regions = split(parts)
    merged_regions = merge(regions)
    extracted_regions = extract(merged_regions, image)
    return extracted_regions
