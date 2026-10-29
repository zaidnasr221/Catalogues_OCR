"""Small source-space rectangle utilities shared by detection and grouping."""
import math


def area(box):
    return max(0, box[2]-box[0]) * max(0, box[3]-box[1])


def intersection(a, b):
    return area([max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])])


def expand(box, margin, bounds):
    return [max(bounds[0], box[0]-margin), max(bounds[1], box[1]-margin),
            min(bounds[2], box[2]+margin), min(bounds[3], box[3]+margin)]


def union(a, b):
    return [min(a[0],b[0]), min(a[1],b[1]), max(a[2],b[2]), max(a[3],b[3])]


def distance(a, b):
    return math.hypot(max(a[0]-b[2],b[0]-a[2],0), max(a[1]-b[3],b[1]-a[3],0))


