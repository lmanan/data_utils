import re
from typing import Tuple

import numpy as np
import numpy.typing as npt


def load_tracklet_csv_data(
    csv_file_name: str,
    delimiter: str = " ",
) -> Tuple[npt.NDArray[np.int64], npt.NDArray[np.int64], npt.NDArray[np.float64]]:
    """
    Load tracklet keypoint data from a space-delimited CSV file.

    The file is expected to have a header row with columns:
        tracklet_id  time  kp0_y  kp0_x  kp1_y  kp1_x  ...  kpN_y  kpN_x

    The number of keypoints is inferred automatically from the header.

    Args:
        csv_file_name (str): Path to the CSV file.
        delimiter (str): Delimiter used in the file. Defaults to ' '.

    Returns:
        Tuple of:
            - tracklet_id: (N,) int array of tracklet IDs
            - time: (N,) int array of time indices
            - keypoints: (N, n_keypoints, 2) float array of (y, x) coordinates
    """
    with open(csv_file_name, "r", encoding="utf-8") as f:
        header_line = f.readline().strip()
    if header_line.startswith("# "):
        header_line = header_line[2:]
    elif header_line.startswith("#"):
        header_line = header_line[1:]
    header = header_line.split(delimiter)

    # Detect keypoint column indices
    kp_pattern = re.compile(r"^kp(\d+)_(y|x)$")
    kp_indices: dict[int, dict[str, int]] = {}
    for col_idx, name in enumerate(header):
        match = kp_pattern.match(name)
        if match:
            kp_num = int(match.group(1))
            coord = match.group(2)
            kp_indices.setdefault(kp_num, {})[coord] = col_idx

    n_keypoints = len(kp_indices)
    if n_keypoints == 0:
        raise ValueError(
            "No keypoint columns found. Expected columns named kp0_y, kp0_x, kp1_y, kp1_x, ..."
        )

    tracklet_col = header.index("tracklet_id")
    time_col = header.index("time")

    data = np.genfromtxt(
        csv_file_name,
        delimiter=delimiter,
        skip_header=1,
        encoding="utf-8",
        comments=None,
    )

    if data.ndim == 1:
        data = data[np.newaxis, :]

    tracklet_id = data[:, tracklet_col].astype(np.int64)
    time = data[:, time_col].astype(np.int64)

    keypoints = np.empty((len(data), n_keypoints, 2), dtype=np.float64)
    for kp_num in sorted(kp_indices.keys()):
        keypoints[:, kp_num, 0] = data[:, kp_indices[kp_num]["y"]]
        keypoints[:, kp_num, 1] = data[:, kp_indices[kp_num]["x"]]

    return tracklet_id, time, keypoints
