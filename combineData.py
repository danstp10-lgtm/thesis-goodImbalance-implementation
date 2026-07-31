from XsensUDPListener import XsensUDPListener 
from TSP_utils import *
from FileSaver import FileSaver
import time
from support_functions import *
import cv2
from triad_openvr import triad_openvr
from collections import deque

rows, columns, between_patch_distance = 27, 19, 15

if __name__ == "__main__":
    # Connect to Patches
    TSP_L = TSPDecoder(port="COM8",rows=rows, columns=columns)
    TSP_R = TSPDecoder(port="COM9",rows=rows, columns=columns) # second touch patch
    MVN = XsensUDPListener(host="127.0.0.1", port=9764)
    saver = FileSaver(output_dir="session_data")
    v = triad_openvr.triad_openvr()
    # v.print_discovered_objects()

    cached_raw_frame = np.zeros(
        (rows, columns * 2 + between_patch_distance), dtype=np.uint8
    )
    display_frame = np.zeros(cached_raw_frame.shape, np.uint8)

    print("Starting sync between Touch Sense Patch 1|2 - 7Hz and Xsens MVN Software - 240Hz")

    xsens_XCoM = None

    # XCoM calc prep
    omega_0 = np.sqrt(9.81/0.6)
    com_history = deque()

    try:
        while True:
            if MVN.new_data_available:
                xsens_data=MVN.get_latest_data()
                xsens_CoM = xsens_data["com"] * 100
                time_sec = xsens_data["timecode"] / 1000.0
                com_history.append((time_sec, xsens_CoM))

                if TSP_L.frame_available and TSP_R.frame_available:
                    hand_segments = xsens_data["hand_segments"]

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
                    print(len(com_history))
                    com_history.clear()                  

                    # Sync coordinate frames
                    TSP_XCoM = [0,0]

                    tracker_TSP_corner = v.devices["tracker_1"].get_pose_euler()
                    tracker_on_body = v.devices["tracker_1"].get_pose_euler()
        
                    # if tracker_on_body is not None and tracker_TSP_corner is not None:
                    #     txt = f"X: {tracker_TSP_corner[0]:.3f} Y: {tracker_TSP_corner[1]:.3f} Z: {tracker_TSP_corner[2]:.3f} | Yaw: {tracker_TSP_corner[3]:.1f} Pitch: {tracker_TSP_corner[4]:.1f} Roll: {tracker_TSP_corner[5]:.1f}"
                    #     print(f"\r{txt}", end="")
                        

                    # print(f"xsens_CoM:{xsens_CoM} | xsens_XCoM:{xsens_XCoM} | TSP_XCoM:{TSP_XCoM}")
                    
                    # Raw TSP data
                    raw_frame_L, raw_frame_R = get_raw_frames(TSP_L,TSP_R)
                    display_frame = np.zeros(cached_raw_frame.shape, np.uint8)                   

                    # determine between patch space
                    # left_hand = hand_segments[0][0:2] * 100
                    # right_hand = hand_segments[1][0:2] * 100
                    # left_right_distance = np.sqrt(np.power(right_hand[0]-left_hand[0],2)+ np.power(right_hand[1]-left_hand[1],2))
                    # between_patch_distance = round((left_right_distance - 2*19*0.8)/0.8)
                    # print(f"patch_dist:{between_patch_distance}")

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
