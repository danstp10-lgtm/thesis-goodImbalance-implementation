import socket
import numpy as np
import struct
import threading
import time

class XsensUDPListener:
    def __init__(self, port=9764, host='127.0.0.1', packet_length=2000):
        self.host=host
        self.port=port
        self.packet_length=packet_length

        self.latest_com=None
        self.segments=None # maybe rename to hands later if used for coord sync
        self.latest_timecode=0.0
        self.new_data_available=False

        self._lock = threading.Lock()
        self._running = True

        self.socket = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
        self.socket.bind((host, port))

        self.thread = threading.Thread(
            target=self._listen_loop,daemon=True
        )
        self.thread.start()

    def _listen_loop(self):
        last_received = None
        last_message_type = None
        last_CoM_readings = [0,0]
        while self._running:
            try:
                data, _ = self.socket.recvfrom(8*self.packet_length)
                if not data:
                    continue
                pos, ori, last_received, timecode, new_packet_flag, last_message_type = self.parse_packet(data, last_received, last_message_type)  
                if new_packet_flag:
                    with self._lock:
                        self.latest_timecode = timecode
                        if last_message_type == 24:
                            self.latest_com = np.asarray([pos[0],pos[1]]) # get x and y axis coordinates
                        elif last_message_type == 2:
                            self.segments = pos
                        self.new_data_available = True
            except Exception as e:
                print(e)
                time.sleep(0.001)

    def get_latest_data(self):
        with self._lock:
            self.new_data_available=False
            return {
                "com":self.latest_com,
                "segments":self.segments,
                "timecode":self.latest_timecode
           }

    def parse_packet(self, message, last_received, last_message_type):
        
        # Header
        if not isinstance(message, (bytes, bytearray)):
            message = bytes(message)
            
        message_id = message[0:6].decode('ascii', errors='ignore')
        try:
            message_type = int(message_id[4:6])
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
            segments = [10,14] # choose which segments to send
            packet_size = 32 
            pos = np.zeros((2, 3))
            ori = np.zeros((2, 4))
            for s in range(segments):
                start = header_length + packet_size*segments[s] 
                floats = struct.unpack('>7f', message[start + 4 : start + packet_size])
                pos[s, :] = floats[0:3]
                ori[s, :] = floats[3:7]
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

        # print(f"i: {message_id}\n"
        #     f"message_type: {message_type}\n"
        #     f"sample_counter: {sample_counter}\n"
        #     f"datagram_counter: {datagram_counter}\n"
        #     f"num_segments: {num_segments}\n"
        #     f"timecode: {timecode:.0f}\n"
        #     f"Data: {pos}\n")

        return pos, ori, sample_counter, timecode, new_packet_flag, message_type

    def close(self):
            """Gracefully closes the socket and stops the UDP listener thread."""
            self._running = False
            try:
                self.socket.close()
            except Exception:
                pass

            if self.thread.is_alive():
                self.thread.join(timeout=1.0)
            print("Xsens UDP Listener closed.")

def main():
    MVN = XsensUDPListener(host="127.0.0.1", port=9764)
    print("Listening for Xsens UDP stream...")

    try:
        while True:
            new_data = MVN.get_latest_data()
            print(new_data)
            time.sleep(0.005)

    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        MVN.close()


if __name__ == "__main__":
    main()

        
