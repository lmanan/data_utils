import re

import numpy as np


def load_tracklet_csv_data(
    csv_file_name: str,
    delimiter: str = " ",
) -> np.ndarray:
    """
    Load tracklet keypoint data from a space-delimited CSV file.

    The file is expected to have a header row with columns:
        tracklet_id  time  kp0_y  kp0_x  kp1_y  kp1_x  ...  kpN_y  kpN_x

    The number of keypoints is inferred automatically from the header.

    Args:
        csv_file_name (str): Path to the CSV file.
        delimiter (str): Delimiter used in the file. Defaults to ' '.

    Returns:
        np.ndarray: Structured array with columns (tracklet_id, time, kp0_y, kp0_x, ..., kpN_y, kpN_x).
    """
    with open(csv_file_name, "r", encoding="utf-8") as f:
        header_line = f.readline().strip()
    if header_line.startswith("# "):
        header_line = header_line[2:]
    elif header_line.startswith("#"):
        header_line = header_line[1:]
    header = header_line.split(delimiter)

    kp_pattern = re.compile(r"^kp(\d+)_(y|x)$")
    kp_col_names = [name for name in header if kp_pattern.match(name)]

    if not kp_col_names:
        raise ValueError(
            "No keypoint columns found. Expected columns named kp0_y, kp0_x, kp1_y, kp1_x, ..."
        )

    kp_fields = [(name, "f8") for name in kp_col_names]
    dtype = np.dtype([("tracklet_id", "i8"), ("time", "i8")] + kp_fields)

    data = np.genfromtxt(
        csv_file_name,
        delimiter=delimiter,
        names=True,
        dtype=dtype,
        encoding="utf-8",
        autostrip=True,
    )

    return data
