import queue
import threading
from pathlib import Path
import numpy as np
import csv

class FileSaver:
    def __init__(self, output_dir="session_data", frames_subdir = "frames", metrics_filename="session_metrics.csv"):
        self.output_dir = Path(output_dir)
        self.frames_dir = self.output_dir / frames_subdir
        self.metrics_file_path = self.output_dir / metrics_filename
        self.output_dir.mkdir(parents=True,exist_ok=True)
        self.frames_dir.mkdir(parents=True, exist_ok=True)

        self.queue = queue.Queue()
        self.frame_counter = 0
        self._running = True
        self._init_csv()

        # start writing loop in a seperate thread
        self.worker_thread = threading.Thread(
            target=self._writer_loop, daemon=True
        )
        self.worker_thread.start()

    def _init_csv(self):
        """Creates the CSV file header if it doesn't already exist."""
        if not self.metrics_file_path.exists():
            with open(
                self.metrics_file_path, mode="w", newline=""
            ) as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow(
                    [
                        "frame_id",
                        "timecode",
                        "cop_x",
                        "cop_y",
                        "cop2bos_dist_cm",
                        "xcom_x",
                        "xcom_y",
                        "xcom_bos_dist_cm",
                    ]
                )

    def save_frame(self,frame,timecode):
        # add frames to queue from data processing script
        self.queue.put(("FRAME",(self.frame_counter, timecode, frame.copy())))
    
    def save_metrics(self, timecode, cop=None, cop2bos_dist=None, xcom=None, xcom2bos_dist=None):
        metrics_payload = {
            "frame_id": self.frame_counter,
            "timecode": timecode,
            "cop": cop,
            "cop2bos_dist": cop2bos_dist,
            "xcom": xcom,
            "xcom2bos_dist": xcom2bos_dist,
        }
        self.queue.put(("METRICS", metrics_payload))

    def increment_frame_count(self):
        self.frame_counter += 1

    def _writer_loop(self):
        with open(self.metrics_file_path, mode="a", newline="") as csv_file:
            writer = csv.writer(csv_file)
            while self._running or not self.queue.empty():
                try:
                    # print(self.queue.get())
                    item_type, payload = self.queue.get(timeout=0.1)

                    if item_type == "FRAME":
                        frame_id, timecode, frame = payload
                        filename = (
                            self.frames_dir
                            / f"frame_{frame_id:06d}_tc_{int(timecode)}.npz"
                        )
                        np.savez_compressed(
                            filename,
                            frame=frame,
                            timecode=timecode,
                            frame_id=frame_id,
                        )

                    elif item_type == "METRICS":
                        data = payload
                        cop = (
                            data["cop"]
                            if data["cop"] is not None
                            else [np.nan, np.nan]
                        )
                        xcom = (
                            data["xcom"]
                            if data["xcom"] is not None
                            else [np.nan, np.nan]
                        )

                        row = [
                            data["frame_id"],
                            data["timecode"],
                            cop[0],
                            cop[1],
                            (
                                data["cop2bos_dist"]
                                if data["cop2bos_dist"] is not None
                                else np.nan
                            ),
                            xcom[0],
                            xcom[1],
                            (
                                data["xcom2bos_dist"]
                                if data["xcom2bos_dist"] is not None
                                else np.nan
                            ),
                        ]
                        writer.writerow(row)
                        csv_file.flush()  # Force write to disk

                    self.queue.task_done()

                except queue.Empty:
                    continue
    
    def close(self):
        self._running = False
        self.queue.join()
        if self.worker_thread.is_alive():
            self.worker_thread.join(timeout=2.0)
        print("Saver queue flushed and session metrics CSV saved.")

