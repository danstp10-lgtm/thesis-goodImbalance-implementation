#!/usr/bin/env python

import cv2
from cv2 import drawContours
import numpy as np
from scipy.ndimage import binary_erosion
from scipy.ndimage import median_filter
from scipy.spatial import ConvexHull

from support_functions import *

# # spacing 8,6mm
pixel_X_length=0.8
pixel_Y_length=0.6

omega_0 = np.sqrt(9.81/0.6)

COLOR_COP = (0, 0, 255)  # Bright Red
COLOR_BOS = (255, 0, 0)  # Bright Blue

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
        # display_frame[CoP_pixel[0]][CoP_pixel[1]] = 255
        cv2.circle(display_frame, (CoP_pixel[1],CoP_pixel[0]), radius=1, color=COLOR_COP, thickness=-1)
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
        cv2.drawContours(display_frame, [BoS_pixel], -1, COLOR_BOS, 1)
        # cv2.polylines(display_frame, [BoS_pixel], isClosed=True, color=(0, 255, 255), thickness=1)
    return BoS_cm

def calculate_XCoM(com_history, latest_CoM, time_sec):
    oldest_time, oldest_com = com_history[0]
    dt = time_sec - oldest_time
    if len(com_history) > 2:
        times = np.array([t for t, _ in com_history])
        positions = np.array([pos for _, pos in com_history])
        t_centered = times - times[0]
        velocity_CoM, _ = np.polyfit(t_centered, positions, deg=1)
        XCoM = latest_CoM + (velocity_CoM / omega_0)
    else:
        velocity_CoM = np.zeros(2)
        XCoM = latest_CoM.copy()
    # print(f"time period: {time_sec}:{oldest_time}={dt} | velocity_CoM: {velocity_CoM} | XCoM: {XCoM}")
    # print(len(com_history))
    return XCoM

def calculate_min_dist(BoS, CoP,XCoM):
    distances_CoP2BoS = []
    distances_XCoM2BoS = []
    min_dist_CoP2BoS = None
    min_dist_XCoM2BoS = None
    if BoS: # if there are any contours
        for i in range(len(BoS)):
            j=i+1
            if j>=len(BoS):
                j=0
            x_1, x_2 = BoS[i][0],BoS[j][0]
            y_1, y_2 = BoS[i][1],BoS[j][1]
            A = y_2 - y_1
            B = x_1 - x_2
            C = x_2*y_1-x_1*y_2
            distances_CoP2BoS.append(np.abs(A*CoP[0]+B*CoP[1]+C)/(np.sqrt(np.power(A,2)+np.power(B,2))))
            distances_XCoM2BoS.append(np.abs(A*XCoM[0]+B*XCoM[1]+C)/(np.sqrt(np.power(A,2)+np.power(B,2))))
        if len(distances_CoP2BoS) > 0 and len(distances_XCoM2BoS) > 0:
            min_dist_CoP2BoS = np.min(distances_CoP2BoS)
            min_dist_XCoM2BoS = np.min(distances_XCoM2BoS)
    return min_dist_CoP2BoS, min_dist_XCoM2BoS

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

        