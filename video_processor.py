import sys

import cv2
from config import INPUT_VIDEO, OUTPUT_VIDEO

NO_DISPLAY = "--no-display" in sys.argv


class VideoProcessor:

    def __init__(self):
        self.cap = cv2.VideoCapture(str(INPUT_VIDEO))

        if not self.cap.isOpened():
            raise FileNotFoundError(
                f"Could not open video: {INPUT_VIDEO}"
            )

        self.width = int(
            self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        )

        self.height = int(
            self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        )

        self.fps = self.cap.get(cv2.CAP_PROP_FPS)

        self.frame_count = int(
            self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
        )

        if self.fps <= 0:
            self.fps = 30

        OUTPUT_VIDEO.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        self.writer = cv2.VideoWriter(
            str(OUTPUT_VIDEO),
            fourcc,
            self.fps,
            (self.width, self.height)
        )

    def print_video_info(self):

        duration = self.frame_count / self.fps

        print("\n========== VIDEO INFORMATION ==========")
        print(f"File         : {INPUT_VIDEO.name}")
        print(f"Resolution   : {self.width} x {self.height}")
        print(f"FPS          : {self.fps:.2f}")
        print(f"Frames       : {self.frame_count}")
        print(f"Duration     : {duration:.2f} seconds")
        print("=======================================\n")

def process_video(self):

        while True:

            success, frame = self.cap.read()

            if not success:
                break

            self.writer.write(frame)

            if not NO_DISPLAY:
                cv2.imshow("Cricket Batting Video", frame)

                key = cv2.waitKey(25) & 0xFF

                if key == ord("q"):
                    break

        self.cap.release()
        self.writer.release()

        if not NO_DISPLAY:
            cv2.destroyAllWindows()

        print("Video processing completed.")
        print(f"Output saved to: {OUTPUT_VIDEO}")


