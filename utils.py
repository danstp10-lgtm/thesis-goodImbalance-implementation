#!/usr/bin/env python

import cv2
from cv2 import drawContours
import numpy as np
from scipy.ndimage import binary_erosion
from scipy.ndimage import median_filter
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation as R

from support_functions import *

# # spacing 8,6mm
pixel_X_length=0.8
pixel_Y_length=0.6

omega_0 = np.sqrt(9.81/0.6)

COLOR_COP = (0, 0, 255)  # Bright Red
COLOR_BOS = (255, 0, 0)  # Bright Blue
COLOR_COM = (0, 255, 0)  # Bright Green
COLOR_XCOM = (0, 255, 255)  # Bright Yellow

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

def process_CoP(raw_frame,display_frame):
    pressure_sum = raw_frame.sum()
    if pressure_sum>0:
        x_grid, y_grid = np.indices(raw_frame.shape)
        CoP_pixel = (int((raw_frame * x_grid).sum() / pressure_sum), int((raw_frame * y_grid).sum() / pressure_sum))
        # cv2.circle(display_frame, (CoP_pixel[1],CoP_pixel[0]), radius=1, color=COLOR_COP, thickness=-1)
        cv2.drawMarker(display_frame,(CoP_pixel[1],CoP_pixel[0]),COLOR_COP,cv2.MARKER_STAR,1,1)
        CoP_cm = (CoP_pixel[0]*pixel_X_length,CoP_pixel[1]*pixel_Y_length)
    else:
        CoP_cm = [0,0]        
    return CoP_cm

def process_BoS(raw_frame, display_frame):
    binary_mask = (raw_frame > 80).astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    BoS_cm = None
    if contours:
        allPts = np.vstack(contours)
        BoS_pixel = cv2.convexHull(allPts,clockwise=False).reshape(-1, 2)
        BoS_cm = [(a * pixel_X_length, b * pixel_Y_length) for a, b in BoS_pixel]
        # Add BoS_pixel to display
        cv2.drawContours(display_frame, [BoS_pixel], -1, COLOR_BOS, 1)
    return BoS_cm

def process_XCoM(com, time_sec, R, t, R_TSP, t_TSP, display_frame):
    com_pos = com[0:3]
    com_vel = com[3:7]
    xsens_XCoM = com_pos + (com_vel / omega_0)

    # Transform Xsens XCoM to TSP coordinates
    TSP_XCoM = transform_Xsens2TSP(xsens_XCoM, R, t, R_TSP, t_TSP ) * 100 # transform XCoM to TSP
    TSP_CoM = transform_Xsens2TSP(com_pos, R, t, R_TSP, t_TSP ) * 100 # transform CoM to TSP

    # Show XCoM on display
    XCoM_pixel = [round(TSP_XCoM[0]/0.8),round(TSP_XCoM[1]/0.6)]
    if 0 < XCoM_pixel[0] < display_frame.shape[1] and 0 < XCoM_pixel[1] < display_frame.shape[0]:
        cv2.drawMarker(display_frame,(XCoM_pixel[0],XCoM_pixel[1]),COLOR_XCOM,cv2.MARKER_STAR,1,1)
        print("XCoM in bounds")

    # Show CoM on display
    CoM_pixel = [round(TSP_CoM[0]/0.8), round(TSP_CoM[1]/0.6)]
    print(f"CoM pixel: {CoM_pixel}")
    if 0 < CoM_pixel[0] < display_frame.shape[1] and 0 < CoM_pixel[1] < display_frame.shape[0]:
        cv2.drawMarker(display_frame,(CoM_pixel[0],CoM_pixel[1]),COLOR_COM,cv2.MARKER_STAR,1,1)
        print("CoM in bounds")
        
    return TSP_XCoM

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
            distances_CoP2BoS.append((A*CoP[0]+B*CoP[1]+C)/(np.sqrt(np.power(A,2)+np.power(B,2))))
            distances_XCoM2BoS.append((A*XCoM[0]+B*XCoM[1]+C)/(np.sqrt(np.power(A,2)+np.power(B,2))))
            min_dist_CoP2BoS = np.min(distances_CoP2BoS) # np.where(distances_CoP2BoS > 0, distances_CoP2BoS, np.inf).argmin()
            min_dist_XCoM2BoS = np.min(distances_XCoM2BoS) # np.where(distances_XCoM2BoS > 0, distances_XCoM2BoS, np.inf).argmin()
            # if min_dist_CoP2BoS == np.inf:
            #     min_val = np.min()   
    return min_dist_CoP2BoS, min_dist_XCoM2BoS

def get_Xsens2Tundra_transforms(xsens_samples, tundra_samples):
    # Compute centroids
    centroid_xsens = np.mean(xsens_samples, axis=0)
    centroid_tundra = np.mean(tundra_samples, axis=0)

    # Bring both to origin
    X = xsens_samples - centroid_xsens
    V = tundra_samples - centroid_tundra

    # Compute the covariance matrix
    H = np.dot(X.T, V)

    # SVD
    U, S, Vt = np.linalg.svd(H)

    # Validate right-handed coordinate system
    if np.linalg.det(np.dot(Vt.T, U.T)) < 0.0:
        Vt[-1, :] *= -1.0

    # Optimal rotation
    R = np.dot(Vt.T, U.T)

    # Optimal translation (depends on R, so computed after it)
    t = centroid_tundra - np.dot(R, centroid_xsens)

    # RMSD
    rmsd = np.sqrt(np.sum(np.square(np.dot(X, R.T) - V)) / xsens_samples.shape[0])

    return R, t, rmsd


def transform_Xsens2TSP(P_xsens, R_xv, t_xv, R_TSP, t_TSP):
    """
    Transforms a 3D point from Xsens space directly to TSP space.
    P_xsense - point in Xsens space
    R_xv - rotation matrix Xsens to Tundra
    t_xv - trainslation Xsens to Tundra
    t_TSP - origin of TSP, marked by third Tundra controller
    R_TSP - rotation of TSP, also get from Tundra controller
    """
    xsens_tundra = (R_xv @ P_xsens) + t_xv # Xsens point to Tundra Space
    xsens_TSP = R_TSP.T @ (xsens_tundra - t_TSP) # Tundra point to TSP Space
    return xsens_TSP
        
def aggragate(data):
    N = len(data)
    if N > 1:
        alpha_decay=0.8
        weights = alpha_decay ** np.arange(N - 1, -1, -1)
        weights /= np.sum(weights)  # Normalize weights
        aggragate_data = np.sum(data * weights[:, None], axis=0)
        # print(f"data: {data} \n into {aggragate_data}")
    else: 
        return data.copy()
    return aggragate_data
