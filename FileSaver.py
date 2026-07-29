import queue
import threading
from pathlib import Path
import numpy as np

class FileSaver:
    def __init__(self, output_dir="recorded_frames"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True,exist_ok=True)

        self.queue = queue.Queue()
        self.frame_counter = 0
        self._running = True

        # start writing loop in a seperate thread
        self.worker_thread = threading.Thread(
            target=self._writer_loop, daemon=True
        )

        self.worker_thread.start()

    def save(self,frame,timecode):
        # add frames to queue from data processing script
        self.queue.put((self.frame_counter, timecode, frame.copy()))
        self.frame_counter += 1

    def _writer_loop(self):
        # write to output_dir while thread is running or there are unsaved frames
        while self._running or not self.queue.empty():
            try:
                frame_id, timecode, frame = self.queue.get(timeout=0.1) # get data from queue
                filename = ( # construct filename
                    self.output_dir
                    / f"frame_{frame_id:06d}_tc_{int(timecode)}.npz"
                )
                # save to file
                np.savez_compressed(
                    filename,
                    frame = frame,
                    timecode = timecode,
                    frame_id = frame_id,
                )

                self.queue.task_done() # announce task done to thread, ensures all items put in the queue get processed
            except queue.Empty:
                continue
    
    def close(self):
        self._running = False
        self.queue.join()

