#!/usr/bin/env python
""" Bounding box
"""
import numpy as np
import cv2

def add_heat(heatmap, bbox_list, weight=1.0):
    """ Add a value to a heatmap
    """
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += weight

    return heatmap

def apply_threshold(heatmap, threshold):
    """ Apply threshold to a heatmap
    """
    heatmap[heatmap <= threshold] = 0
    return heatmap

def draw_labeled_bboxes(image, labels):
    """ Draw bounding boxes on a image
    """
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)),
                (np.max(nonzerox), np.max(nonzeroy)))

        cv2.rectangle(image, bbox[0], bbox[1], (0, 0, 255), 2)

    return image
