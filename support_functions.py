import json
import numpy as np
import serial
import serial.serialutil
import serial.tools.list_ports
import time
import threading


class TSPDecoder:
    """
    TSPDecoder class for handling communication with the TSP device.

    Methods
    -------
    __init__(self, port: str = None, baudrate: int = 921600, rows: int = 27, columns: int = 19)
        Initializes the TSPDecoder instance.

    resync(self) -> None:
        Resynchronizes the TSP serial communication.

    updateFrame(self) -> None:
        Updates the frame data continuously from the TSP device.

    readFrame(self) -> np.ndarray:
        Returns the current frame data.

    getSerialPort(self) -> serial.tools.list_ports_common.ListPortInfo:
        Returns the port/device of the first connected Arduino.

    """

    def __init__(self, port: str = None, baudrate: int = 921600, rows: int = 27, columns: int = 19):
        """
       Initializes the TSPDecoder instance.

       Parameters
       ----------
       port : str, optional
           The serial port to use. If not provided, it finds the first connected Arduino.
       baudrate : int, optional
           Baud rate for serial communication.
       rows : int, optional
           Number of rows in the frame.
       columns : int, optional
           Number of columns in the frame.
       """

        # Initialize TSPDecoder instance with specified rows and columns
        self.rows = rows
        self.columns = columns
        self.frame = np.zeros([rows, columns])

        # If no port is provided, get the first connected Arduino's port
        if not port:
            port = self.getSerialPort()

        # Initialize serial port communication with specified parameters
        self.port = serial.Serial(port, baudrate, timeout=1)

        # Initialize the bool to check serial connection is present
        self.availabool = True

        # Initialize bool to indicate whether there is a new frame available to read
        self.frame_available = False

        # Setup a thread for the frame updating function
        updateThread = threading.Thread(target=self.updateFrame)
        updateThread.daemon = True  # Make the thread dependant on the main program thread, to ensure no thread leak occurs
        updateThread.start()

        # After starting the serial readout, give the TSP some time to calibrate
        print("Calibrating TSP")
        # time.sleep(5)
        print("TSP calibrated, starting datastream")

    def resync(self) -> None:
        """
        Resynchronizes the TSP serial communication.

        Returns
        -------
        None
        """

        antispam = True
        while True:
            # Read a line from the serial port
            buf = self.port.readline()
            try:
                # Decode the last 6 characters of the buffer
                l = buf[-6:].decode()

                # Check for correct frame format or trigger resync
                if (buf.__len__() != 6) or (l != "FRAME\n"):
                    if antispam:
                        print("Resyncing....")
                        antispam = False
                if l == "FRAME\n":
                    break
            except:
                print("Undecodable buffer of length ", len(buf))

    def updateFrame(self) -> None:
        """
        Updates the frame data continuously from the TSP device.

        Returns
        -------
        None
        """

        # Resynchronize TSP communication
        self.resync()

        # Run indefinitely, possible because a Thread was opened
        while True:

            time.sleep(0.0010)  # small delay to break give TSP time to push more data

            # Only try to read frames when the serial object is available
            try:
                
                # Check for lost synchronization and resync if needed
                l = self.port.readline()
                
                img = np.zeros((self.rows, self.columns))
                
                if l.decode()[-4:] == "FR0\n":
                    self.frame = img
                    self.frame_available = True
                    continue
                
                if l.decode()[-6:] != "FRAME\n":
                    print("Lost sync '%s'" % (l.decode()))
                    self.resync()

                # Calculate the frame length
                length = self.rows * self.columns + 1
                
                res =  self.port.read(length)
                length -= len(res)

                # Continue reading until the specified length is reached
                while length != 0:
                    l = self.port.read(length)
                    length -= len(l)
                    res += l

                # Process the received data into the frame
                r = 0
                c = 0
                for v in res[:-1]:
                    img[r][c] = 1.5 * (v)
                    c += 1
                    if c == self.columns:
                        c = 0
                        r += 1

                # Update the frame with clipped and rotated image data
                self.frame = np.clip(np.rot90(img, 2), 0, 255)
                self.frame_available = True
                
                
                self.availabool = True

            # Make the serial flag unavailable if serial is closed
            except serial.serialutil.SerialException:
                self.availabool = False

    def readFrame(self) -> np.array:
        """
        Returns the current frame data.

        Returns
        -------
        np.ndarray
            2D NumPy array representing the frame.
        """
        
        if self.frame_available:
            self.frame_available = False
            return self.frame
        else:
            return None



    def available(self) -> bool:
        """
        Returns the availability of the serial port

        Returns
        -------
        bool
            Boolean representing the availability of the serial port
        """
        return self.availabool

    def getSerialPort(self) -> serial.tools.list_ports_common.ListPortInfo:
        """
        Returns the port/device of the first connected Arduino.

        Raises
        ------
        AssertionError
            If no connected Arduino could be found.

        Returns
        -------
        device : serial.tools.list_ports_common.ListPortInfo
            Full device path.
        """
        # Get all available ports
        ports = list(serial.tools.list_ports.comports())
        device = None

        arduino_port_keywords = [
            "SLAB_USBtoUART",
            "Silicon Labs"
        ]

        # Look through all ports and find the one with a Arduino device
        for p in ports:
            for k in arduino_port_keywords:
                if k in [str(p.manufacturer), str(p.description), str(p.name)]:
                    device = p.device
                    break

        # Return the found port or raise an error
        if not device:
            print("No device found, waiting..")
            time.sleep(2.5)
            self.getSerialPort()
        else:
            return device


class NumpyEncoder(json.JSONEncoder):
    """
    JSON encoder that supports NumPy arrays.

    This class extends the standard JSONEncoder to handle NumPy arrays.
    When encountering a NumPy array in the object to be serialized, it converts
    the array to a Python list using the 'tolist()' method.
    """

    def default(self, o):
        """
        Override the default method of JSONEncoder.

        Parameters
        ----------
        o : object
            The object to be serialized.

        Returns
        -------
        JSON-serializable object
            The serialized version of the object.
        """
        if isinstance(o, np.ndarray):
            return o.tolist()
        return json.JSONEncoder.default(self, o)


def AsciiDecoder(b) -> str:
    """
    Decode key presses from the waitKey function in OpenCV.

    Parameters
    ----------
    b : int
        The keypress code received from the waitKey function.

    Returns
    -------
    str
        Returns the decoded character corresponding to the keypress.
        If the keypress code is '-1', returns "".
    """
    if b == '-1':
        return ""
    # bitmasks the last byte of b and returns decoded character
    return chr(b & 0xFF)