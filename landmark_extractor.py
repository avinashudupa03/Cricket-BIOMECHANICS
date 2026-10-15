import csv
from pathlib import Path

from config import OUTPUT_DATA


class LandmarkExtractor:

    def __init__(self):

        OUTPUT_DATA.mkdir(
            parents=True,
            exist_ok=True
        )

        self.output_file = (
            OUTPUT_DATA / "batting_landmarks.csv"
        )

        self.landmark_names = [
            "nose",
            "left_eye_inner",
            "left_eye",
            "left_eye_outer",
            "right_eye_inner",
            "right_eye",
            "right_eye_outer",
            "left_ear",
            "right_ear",
            "mouth_left",
            "mouth_right",
            "left_shoulder",
            "right_shoulder",
            "left_elbow",
            "right_elbow",
            "left_wrist",
            "right_wrist",
            "left_pinky",
            "right_pinky",
            "left_index",
            "right_index",
            "left_thumb",
            "right_thumb",
            "left_hip",
            "right_hip",
            "left_knee",
            "right_knee",
            "left_ankle",
            "right_ankle",
            "left_heel",
            "right_heel",
            "left_foot_index",
            "right_foot_index"
        ]

    def create_header(self):

        header = [
            "frame",
            "timestamp_ms",
            "tracking_ok",
            "tracking_confidence",
            "interpolated"
        ]

        for name in self.landmark_names:

            header.extend([
                f"{name}_x",
                f"{name}_y",
                f"{name}_z",
                f"{name}_visibility",
                f"{name}_presence"
            ])

        return header

    def extract(
        self,
        frame_number,
        timestamp_ms,
        pose,
        tracking_ok=False,
        tracking_confidence=0.0,
        interpolated=False
    ):

        """Return one CSV row for a frame, including per-frame tracking state.

        tracking_ok / tracking_confidence let downstream steps (and the
        results page) distinguish reliably-tracked batsman poses from
        TRACKING UNCERTAIN frames where the identity lock was withheld.

        interpolated marks frames whose landmark coordinates were filled by
        short-gap temporal interpolation (a continuity estimate, never a
        freshly-detected pose). Downstream consumers can decide whether to
        include interpolated frames in their metrics.

        Each landmark row carries the MediaPipe per-landmark ``visibility``
        and ``presence`` scores (0..1) so angle/feature math can *filter* on
        landmark reliability instead of using every coordinate blindly.
        """
        row = [
            frame_number,
            timestamp_ms,
            int(bool(tracking_ok)),
            round(float(tracking_confidence), 4),
            int(bool(interpolated))
        ]

        if pose is None:

            for _ in self.landmark_names:

                row.extend([
                    "",
                    "",
                    "",
                    "",
                    ""
                ])

        else:

            for landmark in pose:

                row.extend([
                    landmark.x,
                    landmark.y,
                    landmark.z,
                    getattr(landmark, "visibility", ""),
                    getattr(landmark, "presence", "")
                ])

        return row

    def save(self, rows):

        with open(
            self.output_file,
            "w",
            newline="",
            encoding="utf-8"
        ) as file:

            writer = csv.writer(file)

            writer.writerow(
                self.create_header()
            )

            writer.writerows(rows)

        print(
            f"Landmark data saved to: "
            f"{self.output_file}"
        )