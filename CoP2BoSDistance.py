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
TSP_L = TSPDecoder(port="COM8",rows=rows, columns=columns)
# TSP_R = TSPDecoder(port="COM9",rows=rows, columns=columns) # second touch patch

while True:
    if TSP_L.frame_available:
        # Get raw pressure data
        rawFrameL = TSP_L.readFrame().astype(np.uint8)
        # rawFrameR = TSP_R.readFrame().astype(np.uint8)

        # Denoising
        rawFrameL = rawFrameL[:,2:] # remove noise columns
        rawFrameL = cv2.fastNlMeansDenoising(rawFrameL, h=17)
        # rawFrameR = rawFrameR[:, 2:]  # remove noise columns
        # rawFrameR = cv2.fastNlMeansDenoising(rawFrameR, h=17)

        # add empty space between hands
        padding = np.zeros((rows,betweenHandDistance))
        rawFrameL = np.concatenate([rawFrameL, padding], axis=1)

        # simulate a second hand
        flipped = cv2.flip(rawFrameL, 1)
        rawFrameL = np.concatenate([rawFrameL, flipped], axis=1)

        displayFrame = np.zeros(rawFrameL.shape, np.uint8)

        # Calculate CoP
        pressureSum = rawFrameL.sum()
        x_grid, y_grid = np.indices(rawFrameL.shape)
        CoP = (int((rawFrameL * x_grid).sum() / pressureSum), int((rawFrameL * y_grid).sum() / pressureSum))

        # Calculate BoS
        binaryMask = (rawFrameL > 140).astype(np.uint8) * 255
        contours, _ = cv2.findContours(binaryMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            allPts = np.vstack(contours)
            BoS = cv2.convexHull(allPts).squeeze()
            # Add BoS to display
            cv2.drawContours(displayFrame, [BoS], -1, (255, 0, 0), 1)

        # Calculate minimum distance of CoP to BoS boundaries


        # Add CoP to display
        displayFrame[CoP[0]][CoP[1]] = 255

        displayFrame = cv2.resize(displayFrame, (3*224, 2*224)) #resize
        rawFrameL = cv2.resize(rawFrameL/255, (3*224, 2*224)) #resize
        cv2.imshow('displayFrame', displayFrame)
        cv2.imshow('rawFrameL', rawFrameL)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
cv2.destroyAllWindows()



        