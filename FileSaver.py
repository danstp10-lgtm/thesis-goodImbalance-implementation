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

        self.worker_thread = threading.Thread(
            target=self._writer_loop, daemon=True
        )

        self.worker_thread.start()

    def save(self,frame,timecode):
        self.queue.put((self.frame_counter, timecode, frame.copy()))
        self.frame_counter += 1

    def _writer_loop(self):
        while self._running and not self.queue.empty():
            try:
                frame_id, timecode, frame = self.queue.get(timeout=0.1)
                filename = (
                    self.output_dir
                    / f"frame_{frame_id:06d}_tc_{int(timecode)}.npz"
                )

                np.savez_compressed(
                    filename = filename,
                    frame = frame,
                    timecode = timecode,
                    frame_id = frame_id,
                )

                self.queue.task_done()
            except queue.Empty:
                continue
    
    def close(self):
        self._running = False
        self.queue.join()

