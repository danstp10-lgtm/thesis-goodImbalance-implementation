from XsensUDPListener import XsensUDPListener 
from utils import *
from FileSaver import FileSaver
import time
from support_functions import *
import cv2
from triad_openvr import triad_openvr
from collections import deque
import numpy as np

rows, columns, between_patch_distance = 27, 19, 15

if __name__ == "__main__":
    # Connect to Patches
    TSP_L = TSPDecoder(port="COM8",rows=rows, columns=columns)
    TSP_R = TSPDecoder(port="COM11",rows=rows, columns=columns) # second touch patch
    MVN = XsensUDPListener(host="127.0.0.1", port=9764)
    saver = FileSaver(output_dir="session_data",frames_subdir="frames")
    v = triad_openvr.triad_openvr()
    v.print_discovered_objects()
    if "tracker_1" in v.devices:
        v.rename_device("tracker_1","body_tracker")
    if "tracker_2" in v.devices:
        v.rename_device("tracker_2","TSP_corner")

    cached_raw_frame = np.zeros(
        (rows, columns * 2 + between_patch_distance), dtype=np.uint8
    )
    display_frame = np.zeros(cached_raw_frame.shape, np.uint8)
    COLOR_COM = (0, 255, 0)  # Bright Green
    COLOR_XCOM = (0, 255, 255)  # Bright Yellow
    COLOR_HAND = (255, 255, 255)  # Bright White

    print("Starting sync between Touch Sense Patch 1|2 - 7Hz and Xsens MVN Software - 240Hz")

    xsens_XCoM = None
    com_history = deque()

    # Calibration parameters
    calibrated = False
    calibration_samples = 300
    tundra_samples = []
    xsens_samples = []
    ALPHA = 0.01 

    # Tundra Y-up to Z-up matrix
    M_swap = np.array([
        [0, 0, 1],  # Grid X gets Tracker Z
        [1, 0, 0],  # Grid Y gets Tracker X
        [0, 1, 0]   # Height gets Tracker Y
    ])

    try:
        while True:
            if MVN.new_data_available: # On new Xsens data
                xsens_data = MVN.get_latest_data()
                xsens_CoM = xsens_data["com"] 
                time_sec = xsens_data["timecode"] / 1000.0
                xsens_segments = xsens_data["segments"]
                com_history.append((time_sec, xsens_CoM))
                body_tracker_coords = v.devices["body_tracker"].get_pose_quaternion()[0:3] # Tundra tracker

                # Determine transformation parameters: R, t
                if not calibrated:     
                    if len(xsens_samples)==len(tundra_samples)==calibration_samples: # Collect n samples of one Tundra and one Xsens tracker, at roughly the same place
                        xsens_mat = np.array(xsens_samples)
                        tundra_mat = np.array(tundra_samples)
                        R, t, sim_error = get_Xsens2Tundra_transforms(xsens_mat, tundra_mat)
                        print(f"similarity error:{sim_error}")
                        calibrated = True
                    elif body_tracker_coords:
                        # print(f"xsens ({len(xsens_samples)}){xsens_samples} | tundra ({len(tundra_samples)}) : {body_tracker_coords[0:3]}")
                        xsens_samples.append(xsens_segments[0])
                        tundra_samples.append(body_tracker_coords)
                else:
                    # Compensate for drift overtime by applying R,t and comparing to actual tracker on the body 
                    xsens2tundra_segments = ((R @ xsens_segments[0]) + t)        
                    drift_error = body_tracker_coords - xsens2tundra_segments
                    t += ALPHA * drift_error
                    # print(f"drift error:{drift_error}")

                    if TSP_L.frame_available and TSP_R.frame_available: # on TSP available frame compute variables
                        # Raw TSP data
                        raw_frame_L, raw_frame_R = get_raw_frames(TSP_L,TSP_R)
                        display_frame = cv2.cvtColor(cached_raw_frame, cv2.COLOR_GRAY2BGR)                  
                        padding = np.zeros((rows,between_patch_distance)) # add empty space between hands
                        cached_raw_frame = np.concatenate([raw_frame_L, padding,raw_frame_R], axis=1).astype(np.uint8) 

                        # Calculate XCoM, with smoothing
                        xsens_XCoM = calculate_XCoM(com_history, xsens_CoM, time_sec)
                        com_history.clear() # clear history for next timestep                  

                        # Transform Xsens to TSP data
                        TSP_XCoM = [0,0]
                        TSP_corner = np.array(v.devices["TSP_corner"].get_pose_matrix().m) # get tundra pose matrix
                        t_TSP =  TSP_corner[0:3, 3]
                        R_TSP = TSP_corner[0:3, 0:3]
                        TSP_XCoM = transform_Xsens2TSP(xsens_XCoM, R, t, t_TSP, R_TSP ) * 100 # transform to TSP
                        
                        # Check transformation with left hand
                        # xsens2TSP_segments = transform_Xsens2TSP(xsens_segments[0],R, t, t_TSP, R_TSP) * 100
                        # xsens2TSP_segments_pixel = [int(np.round(xsens2TSP_segments[0]/0.8)),int(np.round(xsens2TSP_segments[1]/0.6))] 
                        # # print(f"hand coords {xsens2TSP_segments_pixel} in shape{display_frame.shape}")
                        # if 0 < xsens2TSP_segments_pixel[0] < display_frame.shape[1] and 0 < xsens2TSP_segments_pixel[1] < display_frame.shape[0]:
                        #     # display_frame[xsens2TSP_segments_pixel[0]][xsens2TSP_segments_pixel[1]]= 255
                        #     cv2.drawMarker(
                        #         display_frame,
                        #         (xsens2TSP_segments_pixel[0], xsens2TSP_segments_pixel[1]),
                        #         color=COLOR_HAND,
                        #         markerType=cv2.MARKER_CROSS,
                        #         markerSize=3,
                        #         thickness=1,
                        #     )
                        
                        # Calculate TSP metrics
                        CoP_cm = calculate_CoP(cached_raw_frame, display_frame)
                        BoS_cm = calculate_BoS(cached_raw_frame, display_frame)
                        
                        # Calculate minimum distance of CoP and XCoM to BoS boundaries in cm, margin of stability b
                        min_dist_CoP2BoS, min_dist_XCoM2BoS = calculate_min_dist(BoS_cm, CoP_cm, TSP_XCoM)

                        # Save data
                        saver.save_frame(cached_raw_frame, time_sec)
                        saver.save_metrics(time_sec, CoP_cm, min_dist_CoP2BoS, TSP_XCoM, min_dist_XCoM2BoS)
                        saver.increment_frame_count()

                        # Show XCoM pixel on display
                        XCoM_pixel = [round(TSP_XCoM[0]/0.8),round(TSP_XCoM[1]/0.6)]
                        if 0 < XCoM_pixel[0] < display_frame.shape[1] and 0 < XCoM_pixel[1] < display_frame.shape[0]:
                            cv2.drawMarker(
                                display_frame,
                                (XCoM_pixel[0],XCoM_pixel[1]),
                                color=COLOR_XCOM,
                                markerType=cv2.MARKER_STAR,
                                markerSize=2,
                                thickness=1,
                            )
                            print("XCoM in bounds")
                        else:
                            print("XCoM out of bounds")

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
            cv2.imshow("Synchronized Display", display_resized)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        saver.close()
        MVN.close()
        cv2.destroyAllWindows()
