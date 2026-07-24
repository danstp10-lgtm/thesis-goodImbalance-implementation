import socket
import numpy as np
import struct
import threading

class XsensUDPListener:
    def __init__(self, port=9764, host='127.0.0.1', packetLength=2000):
        self.host=host
        self.port=port
        self.packetLength=packetLength

        self.latestCom=None
        self.latestSegments=None # maybe rename to hands later if used for coord sync
        self.latestTimecode=0.0
        self.newDataAvailable=False

        self._lock = threading.Lock()
        self._running = True

        self.socket = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
        self.socket.bind((host, port))

        self.thread = threading.Thread(
            target=self._listen_loop(),daemon=True
        )
        self.thread.start()

    def _listen_loop(self):
        lastRecieved = None
        lastMessageType = None

        while self._running:
            try:
                data, _ = s.recvfrom(8*packetLength)
                if not data:
                    continue
                pos, ori, lastReceived, timeCode, newPacketFlag, lastMessageType = parse_packet(data,lastReceived, lastMessageType)  
                if newPacketFlag:
                    with self._lock:
                        selfT = timeCode
                        if lastMessageType == 24:
                            print(f"CoM position: {pos}")
                            self.latestCom = pos
                        elif lastMessageType == 2:
                            print(f"Segment position: {pos}")
                            self.latestSegments = pos
            except Exception as e:
                time.sleep(0.001)

    def get_latest_data(self):
        with self._lock:
            self.newDataAvailable=False
            return {
                "com":self.latestCom,
                "hand_segments":self.latestSegments,
                "timecode":selfT
            }

    def parse_packet(message, lastReceived, lastMessageType):
        
        # Header
        if not isinstance(message, (bytes, bytearray)):
            message = bytes(message)
            
        messageId = message[0:6].decode('ascii', errors='ignore')
        try:
            messageType = int(messageId[4:6])
        except ValueError:
            messageType = 0
                
        sampleCounter = struct.unpack('>I', message[6:10])[0] + 1
        datagramCounter = f"{message[10]:b}"
        numSegments = int(message[11])
        timeCode = float(struct.unpack('>I', message[12:16])[0])
        
        pos = np.array([])
        ori = np.array([])

        # Check duplicate package
        if sampleCounter == lastReceived and lastMessageType == messageType:
            newPacketFlag = 0
            return pos, ori, sampleCounter, timeCode, newPacketFlag, messageType
        else:
            newPacketFlag = 1

        # Payload
        headerLength = 24
        if messageType == 2: # Quaternion 23 main segment data
            packetSize = 32 
            pos = np.zeros((2, 3))
            ori = np.zeros((2, 4))
            # Left hand
            start = headerLength + packetSize*14 
            floats = struct.unpack('>7f', message[start + 4 : start + packetSize])
            pos[0, :] = floats[0:3]
            ori[0, :] = floats[3:7]
            # Right hand
            start = headerLength + packetSize*10
            floats = struct.unpack('>7f', message[start + 4 : start + packetSize])
            pos[1, :] = floats[0:3]
            ori[1, :] = floats[3:7]

        elif messageType == 3: # Point Data, likely no Hand
            packetSize = 16
            pos = np.zeros((numSegments, 3))
            for s in range(numSegments):
                start = headerLength + s*packetSize
                floats = struct.unpack('>3f', message[start + 4 : start + packetSize])
                pos[s, :] = floats[0:3]

        elif messageType == 24: # CoM position data
            packetSize = 12
            start = headerLength + packetSize
            floats = struct.unpack('>3f', message[start : start + packetSize])
            pos = floats[0:3]
        else:
            newPacketFlag = 1

        # print(f"messageId: {messageId}\n"
        #     f"messageType: {messageType}\n"
        #     f"sampleCounter: {sampleCounter}\n"
        #     f"datagramCounter: {datagramCounter}\n"
        #     f"numSegments: {numSegments}\n"
        #     f"timeCode: {timeCode:.0f}\n"
        #     f"Data: {pos}\n")

        return pos, ori, sampleCounter, timeCode, newPacketFlag, messageType

    def main():
        MVN = XsensUDPListener()
        data = MVN.get_latest_data()
        
        while data:
            data = s.recv(8*packetLength)
            message = [data,host]
            pos, ori, lastReceived, timeCode, newPacketFlag, lastMessageType = parse_packet(data,lastReceived, lastMessageType)  
            if lastMessageType == 24:
                print(f"CoM position: {pos}")
            elif lastMessageType == 2:
                print(f"Segment position: {pos}")
        s.close()

        
