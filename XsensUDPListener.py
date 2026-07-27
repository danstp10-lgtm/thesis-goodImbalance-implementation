import socket
import numpy as np
import struct
import threading
import time

class XsensUDPListener:
    def __init__(self, port=9764, host='127.0.0.1', packet_length=2000, g = 9.81, l = 0.7):
        self.host=host
        self.port=port
        self.packet_length=packet_length

        self.latest_com=None
        self.latest_xcom=None
        self.latest_com_vel = np.zeros(2)
        self.latest_segments=None # maybe rename to hands later if used for coord sync
        self.latest_timecode=0.0
        self.new_data_available=False

        # For XCoM calc
        self.omega_0 = np.sqrt(g/l)
        self._prev_com = None
        self._prev_timecode = None

        self._lock = threading.Lock()
        self._running = True

        self.socket = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
        self.socket.bind((host, port))

        self.thread = threading.Thread(
            target=self._listen_loop(),daemon=True
        )
        self.thread.start()

    def _listen_loop(self):
        last_recieved = None
        last_message_type = None
        last_CoM_readings = [0,0]

        while self._running:
            try:
                data, _ = s.recvfrom(8*packet_length)
                if not data:
                    continue
                pos, ori, last_received, timecode, new_packet_flag, last_message_type = self.parse_packet(data,last_received, last_message_type)  
                if new_packet_flag:
                    with self._lock:
                        self.latest_timecode = timecode
                        if last_message_type == 24:
                            print(f"CoM position: {pos}")
                            self.latest_com = [pos[0],pos[1]] # get x and y axis coordinates
                            
                            # Calculate XCoM
                            if (self._prev_com is not None
                                and self._prev_timecode is not None):
                                dt = timecode - self._prev_timecode
                                if dt > 0: # Ensure valid time step to prevent division by zero
                                    self.latest_com_vel = (self.latest_com - self._prev_com) / dt
                                    self.latest_xcom = (self.latest_com + (self.latest_com_vel / self.omega_0))
                            else:
                                self.latest_xcom = self.latest_com.copy()

                            # Cache history for differentiation
                            self._prev_com = self.latest_com.copy()
                            self._prev_timecode = timecode

                        elif last_message_type == 2:
                            print(f"Segmen_t position: {pos}")
                            self.latest_segments = pos
                        self.new_data_available = True
            except Exception as e:
                time.sleep(0.001)

    def get_latest_data(self):
        with self._lock:
            self.new_data_available=False
            return {
                "com":self.latest_com,
                "xcom":self.latest_xcom,
                "hand_segmen_ss":self.latest_segments,
                "timecode":self.latest_timecode
           }

    def parse_packet(message, last_received, last_message_type):
        
        # Header
        if not isinstance(message, (bytes, bytearray)):
            message = bytes(message)
            
        message_id = message[0:6].decode('ascii', errors='ignore')
        try:
            message_type = int(i[4:6])
        except ValueError:
            message_type = 0
                
        sample_counter = struct.unpack('>I', message[6:10])[0] + 1
        datagram_counter = f"{message[10]:b}"
        num_segments = int(message[11])
        timecode = float(struct.unpack('>I', message[12:16])[0])
        
        pos = np.array([])
        ori = np.array([])

        # Check duplicate package
        if sample_counter == last_received and last_message_type == message_type:
            new_packet_flag = 0
            return pos, ori, sample_counter, timecode, new_packet_flag, message_type
        else:
            new_packet_flag = 1

        # Payload
        header_length = 24
        if message_type == 2: # Quaternion 23 main segment data
            packet_size = 32 
            pos = np.zeros((2, 3))
            ori = np.zeros((2, 4))
            # Left hand
            start = header_length + packet_size*14 
            floats = struct.unpack('>7f', message[start + 4 : start + packet_size])
            pos[0, :] = floats[0:3]
            ori[0, :] = floats[3:7]
            # Right hand
            start = header_length + packet_size*10
            floats = struct.unpack('>7f', message[start + 4 : start + packet_size])
            pos[1, :] = floats[0:3]
            ori[1, :] = floats[3:7]

        elif message_type == 3: # Point Data, likely no Hand
            packet_size = 16
            pos = np.zeros((num_segments, 3))
            for s in range(num_segments):
                start = header_length + s*packet_size
                floats = struct.unpack('>3f', message[start + 4 : start + packet_size])
                pos[s, :] = floats[0:3]

        elif message_type == 24: # CoM position data
            packet_size = 12
            start = header_length + packet_size
            floats = struct.unpack('>3f', message[start : start + packet_size])
            pos = floats[0:3]
        else:
            new_packet_flag = 1

        # print(f"i: {i}\n"
        #     f"message_type: {message_type}\n"
        #     f"sample_counter: {sample_counter}\n"
        #     f"datagram_counter: {datagram_counter}\n"
        #     f"num_segments: {num_segments}\n"
        #     f"timecode: {timecode:.0f}\n"
        #     f"Data: {pos}\n")

        return pos, ori, sample_counter, timecode, new_packet_flag, message_type

    def main():
        MVN = XsensUDPListener()
        data = MVN.get_latest_data()
        
        while data:
            data = s.recv(8*packet_length)
            message = [data,host]
            pos, ori, last_received, timecode, new_packet_flag, last_message_type = MVN.packet_packet(data,last_received, last_message_type)  
            if last_message_type == 24:
                print(f"CoM position: {pos}")
            elif last_message_type == 2:
                print(f"Segment position: {pos}")
        s.close()

        
