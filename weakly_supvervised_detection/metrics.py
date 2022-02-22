import cv2
import torch
import numpy as np

def loc_accuracy(outputs, labels, gt_boxes, bboxes, iou_threshold=0.5):
    if outputs is not None:
        _, pred = torch.topk(outputs, k=1, dim=1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(labels.view(1, -1).expand_as(pred))
        wrongs = [c == 0 for c in correct.cpu().numpy()][0]

    batch_size = len(gt_boxes)
    gt_known, top1 = 0., 0.
    for i, (gt_box, bbox) in enumerate(zip(gt_boxes, bboxes)):
        iou_score = iou(gt_box, bbox)

        if iou_score >= iou_threshold:
            gt_known += 1.
            if outputs is not None and not wrongs[i]:
                top1 += 1.

    gt_loc = gt_known / batch_size
    top1_loc = top1 / batch_size
    return gt_loc, top1_loc

def iou(box1, box2):
    """box: (xmin, ymin, xmax, ymax)"""
    box1_xmin, box1_ymin, box1_xmax, box1_ymax = box1
    box2_xmin, box2_ymin, box2_xmax, box2_ymax = box2

    inter_xmin = max(box1_xmin, box2_xmin)
    inter_ymin = max(box1_ymin, box2_ymin)
    inter_xmax = min(box1_xmax, box2_xmax)
    inter_ymax = min(box1_ymax, box2_ymax)

    inter_area = (inter_xmax - inter_xmin + 1) * (inter_ymax - inter_ymin + 1)
    box1_area = (box1_xmax - box1_xmin + 1) * (box1_ymax - box1_ymin + 1)
    box2_area = (box2_xmax - box2_xmin + 1) * (box2_ymax - box2_ymin + 1)

    iou = inter_area / (box1_area + box2_area - inter_area).float()
    return iou.item()
