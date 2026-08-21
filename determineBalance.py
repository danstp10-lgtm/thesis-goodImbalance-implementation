from XsensUDPListener import XsensUDPListener 
from utils import *
from FileSaver import FileSaver
import time
from support_functions import *
import cv2
import numpy as np
from collections import deque

rows, columns, between_patch_distance = 27, 19, 15 # TSP parameters

if __name__ == "__main__":
    # Connect to Touch Sensitive Patches
    TSP_L = TSPDecoder(port="COM8",rows=rows, columns=columns)
    TSP_R = TSPDecoder(port="COM11",rows=rows, columns=columns) 
    MVN = XsensUDPListener(host="127.0.0.1", port=9764) # start Xsens listener
    saver = FileSaver(output_dir="recordings",frames_subdir="frames") # initialize file saver 
    cached_raw_frame = np.zeros((rows, columns * 2 + between_patch_distance), dtype=np.uint8)
    # Visualization
    display_frame = np.zeros(cached_raw_frame.shape, np.uint8)
    COLOR_HAND = (255, 255, 255)  # Bright White
    # XCoM variables
    xsens_XCoM = None
    com_history = []
    calibrated = True

    print("Starting sync between Touch Sense Patch 1|2 - 7Hz and Xsens MVN Software - 240Hz")
    try:
        while True:
            if MVN.new_data_available: # On new Xsens data
                xsens_data = MVN.get_latest_data()
                latest_xsens_CoM = xsens_data["com"] 
                time_sec = xsens_data["timecode"] 
                xsens_segments = xsens_data["segments"]
                TSP_corner = xsens_data["tsp_corner"]
                com_history.append((time_sec, latest_xsens_CoM))

                if TSP_L.frame_available and TSP_R.frame_available: # on TSP available frame compute variables
                    # Raw TSP data
                    raw_frame_L, raw_frame_R = get_raw_frames(TSP_L,TSP_R)
                    display_frame = cv2.cvtColor(cached_raw_frame, cv2.COLOR_GRAY2BGR)                  
                    padding = np.zeros((rows,between_patch_distance)) # add empty space between hands
                    cached_raw_frame = np.concatenate([raw_frame_L, padding,raw_frame_R], axis=1).astype(np.uint8) 

                    # Calculate and translate XCoM, with smoothing
                    t_TSP, R_TSP = TSP_corner[0], TSP_corner[1]
                    com_aggragate = aggragate([item[1] for item in com_history][0])
                    # print(f"aggregated {len(com_history)} frames from {com_history[0][0]} - {com_history[-1][0]} into {com_aggragate}")
                    TSP_XCoM, TSP_CoM = process_XCoM(com_aggragate, time_sec, R_TSP, t_TSP, display_frame)
                    com_history.clear() # clear history for next timestep                  
                    
                    # Check transformation with left hand
                    xsens2TSP_segments = transform_Xsens2TSP(xsens_segments, R_TSP, t_TSP) * 100
                    xsens2TSP_segments_pixel = [int(np.round(xsens2TSP_segments[0]/0.8)),int(np.round(xsens2TSP_segments[1]/0.6))] 
                    print(f"test coords {xsens2TSP_segments_pixel} in shape {display_frame.shape}")
                    if 0 < xsens2TSP_segments_pixel[0] < display_frame.shape[1] and 0 < xsens2TSP_segments_pixel[1] < display_frame.shape[0]:
                        cv2.drawMarker(
                            display_frame,
                            (xsens2TSP_segments_pixel[0], xsens2TSP_segments_pixel[1]),
                            color=COLOR_HAND,
                            markerType=cv2.MARKER_CROSS,
                            markerSize=1,
                            thickness=1,
                        )
                    
                    # Calculate TSP metrics
                    CoP_cm = process_CoP(cached_raw_frame, display_frame)
                    BoS_cm = process_BoS(cached_raw_frame, display_frame)
                    
                    # Calculate minimum distance of CoP and XCoM to BoS boundaries in cm, margin of stability b
                    min_dist_CoP2BoS, min_dist_XCoM2BoS, min_dist_CoM2BoS = calculate_min_dist(BoS_cm, CoP_cm, TSP_CoM, TSP_XCoM)

                    # Save data
                    saver.save_frame(cached_raw_frame, time_sec)
                    if min_dist_CoM2BoS > 0:
                        CoM_in_BoS = TSP_CoM
                    else:
                        CoM_in_BoS = None
                    saver.save_metrics(time_sec, CoP_cm, min_dist_CoP2BoS, CoM_in_BoS, TSP_XCoM, min_dist_XCoM2BoS, BoS_cm)
                    saver.increment_frame_count()

                    # Output synchronized packet info
                    # print(f"Time: {time_sec}s | TSP_XCoM: {TSP_XCoM} | CoP (cm): {CoP_cm} | MinDistCoP: {min_dist_CoP2BoS} | MinDistXCoM {min_dist_XCoM2BoS}")
            else:
                time.sleep(0.0005)  # Yield CPU to UDP thread
            
            # Visualization
            display_resized = cv2.resize(
                display_frame,
                (3 * 224, 2 * 224),
                interpolation=cv2.INTER_NEAREST,
            )
            cv2.imshow("TSP live display", display_resized)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        saver.close()
        MVN.close()
        cv2.destroyAllWindows()
