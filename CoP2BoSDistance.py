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
pixelXLength=0.8
pixelYLength=0.6

# Connect to Patches
TSP_L = TSPDecoder(port="COM8",rows=rows, columns=columns)
TSP_R = TSPDecoder(port="COM10",rows=rows, columns=columns) # second touch patch

def getCleanFrames(TSP_L,TSP_R):
    # Get raw pressure data
        rawFrameL = TSP_L.readFrame().astype(np.uint8)
        rawFrameR = TSP_R.readFrame().astype(np.uint8)

        # Denoising
        # rawFrameL = rawFrameL[:,2:] # remove noise columns
        rawFrameL = cv2.fastNlMeansDenoising(rawFrameL, h=10)
        # rawFrameR = rawFrameR[:, 2:]  # remove noise columns
        rawFrameR = cv2.fastNlMeansDenoising(rawFrameR, h=10)
        return rawFrameL, rawFrameR

def calculateCoP(rawFrame):
    pressureSum = rawFrame.sum()
    x_grid, y_grid = np.indices(rawFrame.shape)
    CoP_pixel = (int((rawFrame * x_grid).sum() / pressureSum), int((rawFrame * y_grid).sum() / pressureSum))
    CoP_cm = (CoP_pixel[0]*pixelXLength,CoP_pixel[1]*pixelYLength)
    return CoP_pixel,CoP_cm

def calculateMinDistCoP2BoS(rawFrame, displayFrame,CoP_cm):
    binaryMask = (rawFrame > 80).astype(np.uint8) * 255
    contours, _ = cv2.findContours(binaryMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    minDistCoP2BoS = None
    distancesCoP2BoS = [] 
    if contours:
        allPts = np.vstack(contours)
        BoS_pixel = cv2.convexHull(allPts).squeeze()
        BoS_cm = [(a * pixelXLength, b * pixelYLength) for a, b in BoS_pixel]
        # Add BoS_pixel to display
        cv2.drawContours(displayFrame, [BoS_pixel], -1, (255, 0, 0), 1)
        
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

            distancesCoP2BoS.append(np.abs(A*CoP_cm[0]+B*CoP_cm[1]+C)/(np.sqrt(A**2+B**2)))
        if len(distancesCoP2BoS) > 0:
            minDistCoP2BoS = np.min(distancesCoP2BoS)
    # print(f"distances:{distancesCoP2BoS}")
    # print(f"minimum:{minDistCoP2BoS}")

    return minDistCoP2BoS

while True:
    if TSP_L.frame_available and TSP_R.frame_available:
        rawFrameL, rawFrameR=getCleanFrames(TSP_L,TSP_R)

        # add empty space between hands
        padding = np.zeros((rows,betweenHandDistance))
        rawFrame = np.concatenate([rawFrameL, padding,rawFrameR], axis=1)

        displayFrame = np.zeros(rawFrame.shape, np.uint8)

        # Calculate CoP
        CoP_pixel,CoP_cm=calculateCoP(rawFrame)

        # Calculate BoS
        minDistCoP2BoS = calculateMinDistCoP2BoS(rawFrame,displayFrame,CoP_cm)

        # Add CoP_pixel to display
        displayFrame[CoP_pixel[0]][CoP_pixel[1]] = 255

        displayFrame = cv2.resize(displayFrame, (3*224, 2*224)) #resize
        rawFrame = cv2.resize(rawFrame/255, (3*224, 2*224)) #resize
        cv2.imshow('displayFrame', displayFrame)
        cv2.imshow('rawFrame', rawFrame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
cv2.destroyAllWindows()



        