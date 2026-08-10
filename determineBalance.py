from XsensUDPListener import XsensUDPListener 
from TSP_utils import *
from FileSaver import FileSaver
import time
from support_functions import *
import cv2
from triad_openvr import triad_openvr
from collections import deque
from calibration import *
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
    omega_0 = np.sqrt(9.81/0.6)
    com_history = deque()
    calibrated = False
    calibration_samples = 300
    vive_samples = []
    xsens_samples = []
    ALPHA = 0.01 
    M_swap = np.array([
        [0, 0, 1],  # Grid X gets Tracker Z
        [1, 0, 0],  # Grid Y gets Tracker X
        [0, 1, 0]   # Height gets Tracker Y
    ])

    try:
        while True:
            if MVN.new_data_available:
                xsens_data=MVN.get_latest_data()
                xsens_CoM = xsens_data["com"] 
                time_sec = xsens_data["timecode"] / 1000.0
                xsens_segments = xsens_data["segments"]
                com_history.append((time_sec, xsens_CoM))
                body_tracker_coords = v.devices["body_tracker"].get_pose_quaternion()[0:3] 
                # Determine transformation parameters: R, t
                if not calibrated:     
                    if len(xsens_samples)==len(vive_samples)==calibration_samples: # Collect n samples of one Vive and one Xsens tracker, at roughly the same place
                        xsens_mat = np.array(xsens_samples)
                        vive_mat = np.array(vive_samples)
                        R, t, sim_error = get_Xsens2Vive_transforms(xsens_mat, vive_mat)
                        print(f"similarity error:{sim_error}")
                        calibrated = True
                    elif body_tracker_coords:
                        # print(f"xsens ({len(xsens_samples)}){xsens_samples} | vive ({len(vive_samples)}) : {body_tracker_coords[0:3]}")
                        xsens_samples.append(xsens_segments[0])
                        vive_samples.append(body_tracker_coords)
                else:
                    # Compensate for drift overtime by applying R,t and comparing to actual tracker on the body 
                    xsens2vive_segments = ((R @ xsens_segments[0]) + t)        
                    drift_error = body_tracker_coords - xsens2vive_segments
                    # print(f"drift error:{drift_error}")
                    t += ALPHA * drift_error
                    if TSP_L.frame_available and TSP_R.frame_available:
                        # Raw TSP data
                        raw_frame_L, raw_frame_R = get_raw_frames(TSP_L,TSP_R)
                        display_frame = cv2.cvtColor(cached_raw_frame, cv2.COLOR_GRAY2BGR)                  
                        padding = np.zeros((rows,between_patch_distance)) # add empty space between hands
                        cached_raw_frame = np.concatenate([raw_frame_L, padding,raw_frame_R], axis=1).astype(np.uint8) 

                        # Calculate XCoM, with smoothing
                        oldest_time, oldest_com = com_history[0]
                        dt = time_sec - oldest_time
                        if len(com_history) > 2:
                            times = np.array([t for t, _ in com_history])
                            positions = np.array([pos for _, pos in com_history])
                            t_centered = times - times[0]
                            velocity_CoM, _ = np.polyfit(t_centered, positions, deg=1)
                            xsens_XCoM = xsens_CoM + (velocity_CoM / omega_0)
                        else:
                            velocity_CoM = np.zeros(2)
                            xsens_XCoM = xsens_CoM.copy()
                        # print(f"time period: {time_sec}:{oldest_time}={dt} | velocity_CoM: {velocity_CoM} | XCoM: {xsens_XCoM}")
                        # print(len(com_history))
                        com_history.clear() # clear history for next timestep                  

                        # Transform Xsens to TSP data
                        TSP_XCoM = [0,0]
                        TSP_corner = np.array(v.devices["TSP_corner"].get_pose_matrix().m) # get vive pose matrix
                        t_TSP =  TSP_corner[0:3, 3]
                        R_TSP = TSP_corner[0:3, 0:3]
                        TSP_XCoM = transform_Xsens2TSP(xsens_XCoM, R, t, t_TSP, R_TSP ) * 100 # transform to TSP
                        
                        # Check transformation with left hand
                        xsens2TSP_segments = transform_Xsens2TSP(xsens_segments[0],R, t, t_TSP, R_TSP) * 100
                        xsens2TSP_segments_pixel = [int(np.round(xsens2TSP_segments[0]/0.8)),int(np.round(xsens2TSP_segments[1]/0.6))] 
                        # print(f"hand coords {xsens2TSP_segments_pixel} in shape{display_frame.shape}")
                        if 0 < xsens2TSP_segments_pixel[0] < display_frame.shape[1] and 0 < xsens2TSP_segments_pixel[1] < display_frame.shape[0]:
                            # display_frame[xsens2TSP_segments_pixel[0]][xsens2TSP_segments_pixel[1]]= 255
                            cv2.drawMarker(
                                display_frame,
                                (xsens2TSP_segments_pixel[0], xsens2TSP_segments_pixel[1]),
                                color=COLOR_HAND,
                                markerType=cv2.MARKER_CROSS,
                                markerSize=3,
                                thickness=1,
                            )
                        
                        # Calculate TSP metrics
                        CoP_cm = calculate_CoP(cached_raw_frame, display_frame)
                        BoS_cm = calculate_BoS(cached_raw_frame, display_frame)
                        
                        # Calculate minimum distance of CoP and XCoM to BoS boundaries in cm, margin of stability b
                        distances_CoP2BoS = []
                        distances_XCoM2BoS = []
                        if BoS_cm: # if there are any contours
                            for i in range(len(BoS_cm)):
                                j=i+1
                                if j>=len(BoS_cm):
                                    j=0
                                x_1, x_2 = BoS_cm[i][0],BoS_cm[j][0]
                                y_1, y_2 = BoS_cm[i][1],BoS_cm[j][1]
                                A = y_2 - y_1
                                B = x_1 - x_2
                                C = x_2*y_1-x_1*y_2
                                distances_CoP2BoS.append(np.abs(A*CoP_cm[0]+B*CoP_cm[1]+C)/(np.sqrt(np.power(A,2)+np.power(B,2))))
                                distances_XCoM2BoS.append(np.abs(A*TSP_XCoM[0]+B*TSP_XCoM[1]+C)/(np.sqrt(np.power(A,2)+np.power(B,2))))
                            if len(distances_CoP2BoS) > 0:
                                min_dist_CoP2BoS = np.min(distances_CoP2BoS)
                                min_dist_XCoM2BoS = np.min(distances_XCoM2BoS)
                                saver.save_metrics(time_sec, CoP_cm, min_dist_CoP2BoS, TSP_XCoM, min_dist_XCoM2BoS)
                        
                        # Save data
                        saver.save_frame(cached_raw_frame, time_sec)
                        saver.increment_frame_count()

                        # Show XCoM pixel on display
                        XCoM_pixel = [round(TSP_XCoM[0]/0.8),round(TSP_XCoM[1]/0.6)]
                        if 0 < XCoM_pixel[0] < display_frame.shape[0] and 0 < XCoM_pixel[1] < display_frame.shape[1]:
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
            # raw_resized = cv2.resize(
            #     cached_raw_frame,
            #     (3 * 224, 2 * 224),
            #     interpolation=cv2.INTER_NEAREST,
            # )
            cv2.imshow("Synchronized Display", display_resized)
            # cv2.imshow("Raw frame", raw_resized)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        saver.close()
        MVN.close()
        cv2.destroyAllWindows()
