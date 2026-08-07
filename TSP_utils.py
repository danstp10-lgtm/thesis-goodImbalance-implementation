#!/usr/bin/env python

import serial
import cv2
from cv2 import drawContours
import numpy as np
import skimage
from scipy.ndimage import binary_erosion
from scipy.ndimage import median_filter
from scipy.spatial import ConvexHull

from support_functions import *

# # spacing 8,6mm
pixel_X_length=0.8
pixel_Y_length=0.6

def get_raw_frames(TSP_L,TSP_R):
    # Get raw pressure data
        raw_frame_L = TSP_L.readFrame().astype(np.uint8)
        raw_frame_R = TSP_R.readFrame().astype(np.uint8)

        # Denoising
        raw_frame_L = cv2.fastNlMeansDenoising(raw_frame_L, h=10)
        raw_frame_R = cv2.fastNlMeansDenoising(raw_frame_R, h=10)
        # raw_frame_L = median_filter(raw_frame_L, size=4)
        # raw_frame_R = median_filter(raw_frame_R, size=4)
        return raw_frame_L, raw_frame_R

def calculate_CoP(raw_frame,display_frame):
    pressure_sum = raw_frame.sum()
    if pressure_sum>0:
        x_grid, y_grid = np.indices(raw_frame.shape)
        CoP_pixel = (int((raw_frame * x_grid).sum() / pressure_sum), int((raw_frame * y_grid).sum() / pressure_sum))
        display_frame[CoP_pixel[0]][CoP_pixel[1]] = 255
        CoP_cm = (CoP_pixel[0]*pixel_X_length,CoP_pixel[1]*pixel_Y_length)
    else:
       CoP_cm = [0,0]        
    return CoP_cm

def calculate_BoS(raw_frame, display_frame):
    binary_mask = (raw_frame > 80).astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    BoS_cm = None
    if contours:
        allPts = np.vstack(contours)
        BoS_pixel = cv2.convexHull(allPts).reshape(-1, 2)
        BoS_cm = [(a * pixel_X_length, b * pixel_Y_length) for a, b in BoS_pixel]
        # Add BoS_pixel to display
        cv2.drawContours(display_frame, [BoS_pixel], -1, (255, 0, 0), 1)

    return BoS_cm

def main():
    rows, columns, between_hand_distance = 27, 19, 15
    while True:
        if TSP_L.frame_available and TSP_R.frame_available:
            raw_frame_L, raw_frame_R=get_raw_frames(TSP_L,TSP_R)

            # add empty space between hands
            padding = np.zeros((rows,between_hand_distance))
            raw_frame = np.concatenate([raw_frame_L, padding,raw_frame_R], axis=1)

            display_frame = np.zeros(raw_frame.shape, np.uint8)

            # Calculate CoP
            CoP_cm = calculate_CoP(raw_frame)

            # Calculate BoS
            BoS = calculate_BoS(raw_frame,display_frame)

            # Add CoP_pixel to display
            display_frame[CoP_pixel[0]][CoP_pixel[1]] = 255

            display_frame = cv2.resize(display_frame, (3*224, 2*224)) #resize
            raw_frame = cv2.resize(raw_frame/255, (3*224, 2*224)) #resize
            cv2.imshow('display_frame', display_frame)
            cv2.imshow('raw_frame', raw_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

        