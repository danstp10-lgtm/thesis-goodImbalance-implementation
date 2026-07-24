#!/usr/bin/env python

import serial
import cv2
from cv2 import drawContours
import numpy as np
import skimage
from scipy.ndimage import binary_erosion
from scipy.spatial import ConvexHull

from support_functions import *

rows, columns, betweenHandDistance = 27, 19, 15
# spacing 8,6mm
pixel_X_length=0.8
pixel_Y_length=0.6

# Connect to Patches
TSP_L = TSPDecoder(port="COM8",rows=rows, columns=columns)
TSP_R = TSPDecoder(port="COM10",rows=rows, columns=columns) # second touch patch

def get_raw_frames(TSP_L,TSP_R):
    # Get raw pressure data
        raw_frame_L = TSP_L.readFrame().astype(np.uint8)
        raw_frame_R = TSP_R.readFrame().astype(np.uint8)

        # Denoising
        # raw_frame_L = raw_frame_L[:,2:] # remove noise columns
        raw_frame_L = cv2.fastNlMeansDenoising(raw_frame_L, h=10)
        # raw_frame_R = raw_frame_R[:, 2:]  # remove noise columns
        raw_frame_R = cv2.fastNlMeansDenoising(raw_frame_R, h=10)
        return raw_frame_L, raw_frame_R

def calculate_CoP(raw_frame):
    pressure_sum = raw_frame.sum()
    if pressure_sum>0:
        x_grid, y_grid = np.indices(raw_frame.shape)
        CoP_pixel = (int((raw_frame * x_grid).sum() / pressure_sum), int((raw_frame * y_grid).sum() / pressure_sum))
        CoP_cm = (CoP_pixel[0]*pixel_X_length,CoP_pixel[1]*pixel_Y_length)
    else:
        CoP_pixel, CoP_cm = [0,0],[0,0]
    return CoP_pixel,CoP_cm

def calculate_min_dist_CoP2BoS(raw_frame, display_frame,CoP_cm):
    binary_mask = (raw_frame > 80).astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_dist_CoP2BoS = None
    distances_CoP2BoS = [] 
    if contours:
        allPts = np.vstack(contours)
        BoS_pixel = cv2.convexHull(allPts).reshape(-1, 2)
        BoS_cm = [(a * pixel_X_length, b * pixel_Y_length) for a, b in BoS_pixel]
        # Add BoS_pixel to display
        cv2.drawContours(display_frame, [BoS_pixel], -1, (255, 0, 0), 1)
        
        # Calculate minimum distance of CoP to BoS boundaries in cm
        for i in range(len(BoS_cm)):
            # \(A = y_2 - y_1\)\(B = x_1 - x_2\)\(C = (x_2 \times y_1) - (x_1 \times y_2)\)
            j=i+1
            if j>=len(BoS_cm):
                j=0
            x_1, x_2 = BoS_cm[i][0],BoS_cm[j][0]
            y_1, y_2 = BoS_cm[i][1],BoS_cm[j][1]
            A = y_2 - y_1
            B = x_1 - x_2
            C = x_2*y_1-x_1*y_2

            distances_CoP2BoS.append(np.abs(A*CoP_cm[0]+B*CoP_cm[1]+C)/(np.sqrt(A**2+B**2)))
        if len(distances_CoP2BoS) > 0:
            min_dist_CoP2BoS = np.min(distances_CoP2BoS)
    # print(f"distances:{distances_CoP2BoS}")
    # print(f"minimum:{min_dist_CoP2BoS}")

    return min_dist_CoP2BoS

while True:
    if TSP_L.frame_available and TSP_R.frame_available:
        raw_frame_L, raw_frame_R=get_raw_frames(TSP_L,TSP_R)

        # add empty space between hands
        padding = np.zeros((rows,betweenHandDistance))
        raw_frame = np.concatenate([raw_frame_L, padding,raw_frame_R], axis=1)

        display_frame = np.zeros(raw_frame.shape, np.uint8)

        # Calculate CoP
        CoP_pixel,CoP_cm=calculate_CoP(raw_frame)

        # Calculate BoS
        min_dist_CoP2BoS = calculate_min_dist_CoP2BoS(raw_frame,display_frame,CoP_cm)

        # Add CoP_pixel to display
        display_frame[CoP_pixel[0]][CoP_pixel[1]] = 255

        display_frame = cv2.resize(display_frame, (3*224, 2*224)) #resize
        raw_frame = cv2.resize(raw_frame/255, (3*224, 2*224)) #resize
        cv2.imshow('display_frame', display_frame)
        cv2.imshow('raw_frame', raw_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
cv2.destroyAllWindows()



        