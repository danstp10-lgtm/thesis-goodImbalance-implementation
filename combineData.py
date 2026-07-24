import XsensUDPListener
import CoP2BoSDistance

rows, columns, between_hand_distance = 27, 19, 15
# spacing 8,6mm
pixelXLength=0.8
pixelYLength=0.6

if name == "__main__":
    # Connect to Patches
    TSP_L = TSPDecoder(port="COM8",rows=rows, columns=columns)
    TSP_R = TSPDecoder(port="COM10",rows=rows, columns=columns) # second touch patch
    MVN = XsensUDPListener(host="127.0.0.1", port=9764)

    cached_raw_frame = np.zeros(
        (rows, columns * 2 + between_hand_distance), dtype=np.uint8
    )

    print("Starting sync between Touch Sense Patch 1|2 - 7Hz and Xsens MVN Software - 240Hz")

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
            hand_segments = xsens_data["hand_segments"]
            timecode = xsens_data["timecode"]
