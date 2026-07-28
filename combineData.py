import XsensUDPListener
import CoP2BoSDistance
import FileSaver
import time

rows, columns, between_patch_distance = 27, 19, 15

if name == "__main__":
    # Connect to Patches
    TSP_L = TSPDecoder(port="COM8",rows=rows, columns=columns)
    TSP_R = TSPDecoder(port="COM10",rows=rows, columns=columns) # second touch patch
    MVN = XsensUDPListener(host="127.0.0.1", port=9764)
    saver = FileSaver(output_dir="session_data")

    cached_raw_frame = np.zeros(
        (rows, columns * 2 + between_patch_distance), dtype=np.uint8
    )

    print("Starting sync between Touch Sense Patch 1|2 - 7Hz and Xsens MVN Software - 240Hz")

    try:
        while True:
            if MVN.new_data_available:
                xsens_data=get_latest_data()

            if TSP_L.frame_available and TSP_R.frame_available:
                # latest Xsens data
                xsens_frame_CoM = xsens_data["com"]
                XCoM = xsens_data["xcom"]
                hand_segments = xsens_data["hand_segments"]
                timecode = xsens_data["timecode"]

                

                # Sync coordinate frames
                left_hand = hand_segments[0]
                right_hand = hand_segments[1]
                TSP_frame_CoM = xsens_frame_CoM - left_hand
                
                # Raw TSP data
                raw_frame_L, raw_frame_R = get_clean_frames(TSP_L,TSP_R)

                # determine between patch space
                between_patch_distance = round(np.sqrt(np.power(right_hand[0]-left_hand[0],2)+ np.power(right_hand[1]-left_hand[1],2)) - 19*0.8)

                # add empty space between hands
                padding = np.zeros((rows,between_patch_distance))
                cached_raw_frame = np.concatenate([raw_frame_L, padding,raw_frame_R], axis=1)

                display_frame = np.zeros(cached_raw_frame.shape, np.uint8)

                # Calculate CoP
                CoP_pixel,CoP_cm=calculate_CoP(cached_raw_frame)

                # Calculate BoS
                BoS_cm = calculate_BoS(cached_raw_frame,display_frame)
                
                # Calculate minimum distance of CoP and CoM to BoS boundaries in cm
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
                        distances_XCoM2BoS.append(np.abs(A*TSP_frame_CoM[0]+B*TSP_frame_CoM[1]+C)/(np.sqrt(np.power(A,2)+np.power(B,2))))
                    if len(distances_CoP2BoS) > 0:
                        min_dist_CoP2BoS = np.min(distances_CoP2BoS)
                        min_dist_XCoM2BoS = np.min(distances_XCoM2BoS)
                    print(f"distances:{distances_CoP2BoS}")
                    print(f"minimum:{min_dist_CoP2BoS}")

                # Show CoP pixel on display
                if cached_raw_frame.sum()>0:
                    cv2.circle(
                        display_frame,
                        (CoP_pixel[1], CoP_pixel[0]),
                        radius=1,
                        color=(0, 0, 255),
                        thickness=-1,
                    )
                
                # Output synchronized packet info
                # print(f"Time: {timecode:.2f}s | xsens_frame_CoM: {com_pos} | CoP (cm): {CoP_cm} | MinDist: {minDistCoP2BoS}")

                # save raw complete frame to file
                saver.save(cached_raw_frame, timecode)

                # Visualization
                display_resized = cv2.resize(
                    display_frame,
                    (3 * 224, 2 * 224),
                    interpolation=cv2.INTER_NEAREST,
                )
                cv2.imshow("Synchronized Display", display_resized)
            else:
                time.sleep(0.0005)  # Yield CPU to UDP thread

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        saver.close()
        cv2.destroyAllWindows()
