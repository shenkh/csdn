# -*- utf-8 -*-
import numpy as np


def nms(boxes, threshold):
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    scores = boxes[:, 4]
    areas = (x2 - x1) * (y2 - y1)
    order = np.argsort(scores)[::-1]
    keep = []
    while len(order) > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[order[1:]] + areas[order[0]] - inter)
        indics = np.where(iou < threshold)[0]
        order = order[indics+1]
    return keep


def soft_nms(boxes, iou_threshold):
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    scores = boxes[:, 4].copy()
    areas = (x2 - x1) * (y2 - y1)
    keep = []
    while True:
        order = np.argsort(scores)[::-1]
        order = order[scores[order] > 0.3]
        if len(order) == 0:
            break
        i = order[0]
        keep.append(i)
        scores[i] = 0

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[order[1:]] + areas[order[0]] - inter)

        indics = np.where(iou >= iou_threshold)[0]
        src_indics = order[indics+1]
        scores[src_indics] *= (1 - iou[indics])
    return boxes[keep]


if __name__ == "__main__":
    np.random.seed(3)
    xy1 = np.random.randint(1, 5, (10, 2), np.uint8)
    xy2 = np.random.randint(15, 20, (10, 2), np.uint8)
    scores = np.random.rand(10, 1)
    boxes = np.concatenate((xy1, xy2, scores), axis=1)
    print(boxes)
    res = nms(boxes, 0.7)
    print(boxes[res])
    res = soft_nms(boxes, 0.7)
    print(res)

