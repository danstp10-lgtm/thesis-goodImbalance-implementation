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
    TSP_R = TSPDecoder(port="COM9",rows=rows, columns=columns) # second touch patch
    MVN = XsensUDPListener(host="127.0.0.1", port=9764)
    saver = FileSaver(output_dir="session_data")
    v = triad_openvr.triad_openvr()
    # v.print_discovered_objects()
    if "tracker_1" in v.devices:
        v.rename_device("tracker_1","body_tracker")
    # if "tracker_2" in v.devices:
        # v.rename_device("tracker_2","TSP_corner")

    cached_raw_frame = np.zeros(
        (rows, columns * 2 + between_patch_distance), dtype=np.uint8
    )
    display_frame = np.zeros(cached_raw_frame.shape, np.uint8)

    print("Starting sync between Touch Sense Patch 1|2 - 7Hz and Xsens MVN Software - 240Hz")

    xsens_XCoM = None
    omega_0 = np.sqrt(9.81/0.6)
    com_history = deque()
    calibrated = False
    vive_samples = np.empty((1,3))
    xsens_samples = np.empty((1,3))

    try:
        while True:
            if MVN.new_data_available:
                xsens_data=MVN.get_latest_data()
                xsens_CoM = xsens_data["com"] * 100
                time_sec = xsens_data["timecode"] / 1000.0
                xsens_segments = xsens_data["segments"]
                com_history.append((time_sec, xsens_CoM))

                # Determine transformation parameters: R, t
                if not calibrated:        
                    body_tracker = v.devices["body_tracker"].get_pose_euler()
                    if len(xsens_samples)==len(vive_samples)==300: # Collect 300 samples of one Vive and one Xsens tracker
                        R, t, error = get_Xsens2Vive_transforms(xsens_samples, vive_samples)
                        print(error)
                        calibrated = True
                    elif body_tracker:
                        # print(f"xsens ({len(xsens_samples)}){xsens_samples} | vive ({len(vive_samples)}) : {body_tracker[0:3]}")
                        xsens_samples = np.vstack([xsens_samples, xsens_segments[0]])
                        vive_samples = np.vstack([vive_samples, body_tracker[0:3]])

                else:
                    if TSP_L.frame_available and TSP_R.frame_available:

                        # Calculate XCoM, with smoothing
                        oldest_time, oldest_com = com_history[0]
                        dt = time_sec - oldest_time
                        if len(com_history) > 2:
                            times = np.array([t for t, _ in com_history])
                            positions = np.array([pos for _, pos in com_history])
                            t_centered = times - times[0]
                            slopes, _ , _ = np.polyfit(t_centered, positions, deg=2)
                            velocity_CoM = slopes  
                            xsens_XCoM = xsens_CoM + (velocity_CoM / omega_0)
                        else:
                            velocity_CoM = np.zeros(2)
                            xsens_XCoM = xsens_CoM.copy()
                        # print(f"time period: {time_sec}:{oldest_time}={dt} | velocity_CoM: {velocity_CoM} | XCoM: {xsens_XCoM}")
                        # print(len(com_history))
                        com_history.clear()                  

                        # Transform Xsens to TSP data
                        TSP_XCoM = [0,0]
                        # TSP_corner = v.devices["TSP_corner"].get_pose_euler()
                        # TSP_XCoM = transform_Xsens2TSP(xsens_XCoM,R,t,TSP_corner[0:3],TSP_corner[3:6])
                        
                        # Raw TSP data
                        raw_frame_L, raw_frame_R = get_raw_frames(TSP_L,TSP_R)
                        display_frame = np.zeros(cached_raw_frame.shape, np.uint8)                   

                        # add empty space between hands
                        padding = np.zeros((rows,between_patch_distance))
                        cached_raw_frame = np.concatenate([raw_frame_L, padding,raw_frame_R], axis=1).astype(np.uint8)
                        
                        # Calculate CoP
                        CoP_cm = calculate_CoP(cached_raw_frame, display_frame)

                        # Calculate BoS
                        BoS_cm = calculate_BoS(cached_raw_frame, display_frame)
                        
                        # Calculate minimum distance of CoP and XCoM to BoS boundaries in cm, margin of stability b
                        distances_CoP2BoS = []
                        distances_XCoM2BoS = []
                        if BoS_cm:
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
                                
                                distances_CoP2BoS.append(np.abs(A*CoP_cm[0]+B*CoP_cm[1]+C)/(np.sqrt(np.power(A,2)+np.power(B,2))))
                                distances_XCoM2BoS.append(np.abs(A*TSP_XCoM[0]+B*TSP_XCoM[1]+C)/(np.sqrt(np.power(A,2)+np.power(B,2))))
                            if len(distances_CoP2BoS) > 0:
                                min_dist_CoP2BoS = np.min(distances_CoP2BoS)
                                min_dist_XCoM2BoS = np.min(distances_XCoM2BoS)

                        # Show XCoM pixel on display
                        # XCoM_pixel = [round(TSP_XCoM[0]/0.8),round(TSP_XCoM[1]/0.6)]
                        # if 0 < XCoM_pixel[0] < display_frame.shape[0] and 0 < XCoM_pixel[1] < display_frame.shape[1]:
                        #     display_frame[XCoM_pixel[0]][XCoM_pixel[1]] = 255
                        # else:
                        #     print("XCoM out of bounds")

                        # Output synchronized packet info
                        # print(f"Time: {time_sec}s | TSP_XCoM: {TSP_XCoM} | CoP (cm): {CoP_cm} | MinDistCoP: {min_dist_CoP2BoS} | MinDistXCoM {min_dist_XCoM2BoS}")
                        saver.save(cached_raw_frame, time_sec)
            else:
                time.sleep(0.0005)  # Yield CPU to UDP thread
            
            # Visualization
            display_resized = cv2.resize(
                display_frame,
                (3 * 224, 2 * 224),
                interpolation=cv2.INTER_NEAREST,
            )
            raw_resized = cv2.resize(
                cached_raw_frame,
                (3 * 224, 2 * 224),
                interpolation=cv2.INTER_NEAREST,
            )
            cv2.imshow("Synchronized Display", display_resized)
            cv2.imshow("raw frame",raw_resized)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        saver.close()
        MVN.close()
        cv2.destroyAllWindows()
