import Xsens2pythonRTstream
import CoP2BoSDistance

host = '127.0.0.1'
port = 9764
packetLength = 2000
lastReceived = None
lastMessageType = None
rows, columns, betweenHandDistance = 27, 19, 15
# spacing 8,6mm
pixelXLength=0.8
pixelYLength=0.6

# Listen to NVM network streamer
s = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
s.bind((host, port))
data, addr = s.recvfrom(8*packetLength)

# Connect to Patches
TSP_L = TSPDecoder(port="COM8",rows=rows, columns=columns)
TSP_R = TSPDecoder(port="COM10",rows=rows, columns=columns) # second touch patch

while data:
    data = s.recv(8*packetLength)
    message = [data,host]
    pos, ori, lastReceived, timeCode, newPacketFlag, lastMessageType = parse_position_packet(data,lastReceived, lastMessageType)  
    # if lastMessageType == 24:
    #     print(f"CoM position: {pos}")
    # elif lastMessageType == 2:
    #     print(f"Segment position: {pos}")
s.close()

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
