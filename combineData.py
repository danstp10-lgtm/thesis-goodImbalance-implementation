import XsensUDPListener
import CoP2BoSDistance
import FileSaver
import time

rows, columns, between_hand_distance = 27, 19, 15

if name == "__main__":
    # Connect to Patches
    TSP_L = TSPDecoder(port="COM8",rows=rows, columns=columns)
    TSP_R = TSPDecoder(port="COM10",rows=rows, columns=columns) # second touch patch
    MVN = XsensUDPListener(host="127.0.0.1", port=9764)
    saver = FileSaver(output_dir="session_data")

    cached_raw_frame = np.zeros(
        (rows, columns * 2 + between_hand_distance), dtype=np.uint8
    )

    print("Starting sync between Touch Sense Patch 1|2 - 7Hz and Xsens MVN Software - 240Hz")

    try:
        while True:
            if MVN.new_data_available:
                xsens_data=get_latest_data()

            if TSP_L.frame_available and TSP_R.frame_available:
                raw_frame_L, raw_frame_R = get_clean_frames(TSP_L,TSP_R)

                # add empty space between hands
                padding = np.zeros((rows,between_hand_distance))
                cached_raw_frame = np.concatenate([raw_frame_L, padding,raw_frame_R], axis=1)

                display_frame = np.zeros(cached_raw_frame.shape, np.uint8)

                # Calculate CoP
                CoP_pixel,CoP_cm=calculate_CoP(cached_raw_frame)

                # Calculate BoS
                min_dist_CoP2BoS = calculate_min_dist_CoP2BoS(cached_raw_frame,display_frame,CoP_cm)

                if cached_raw_frame.sum()>0:
                    cv2.circle(
                        display_frame,
                        (CoP_pixel[1], CoP_pixel[0]),
                        radius=1,
                        color=(0, 0, 255),
                        thickness=-1,
                    )
                
                # latest Xsens data
                CoM = xsens_data["com"]
                XCoM = xsens_data["xcom"]
                hand_segments = xsens_data["hand_segments"]
                timecode = xsens_data["timecode"]

                # Output synchronized packet info
                # print(f"Time: {timecode:.2f}s | CoM: {com_pos} | CoP (cm): {CoP_cm} | MinDist: {minDistCoP2BoS}")

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
