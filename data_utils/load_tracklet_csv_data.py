import re

import numpy as np


def load_tracklet_csv_data(
    csv_file_name: str,
    delimiter: str = " ",
) -> np.ndarray:
    """
    Load tracklet keypoint data from a space-delimited CSV file.

    Accepts both the minimal format (tracklet_id, time/t, kp*) and the full
    detection format (group, id, t, y, x, parent_id, tracklet_id, kp*). Extra
    columns are silently ignored; only tracklet_id, the time column, and all
    kp{i}_y / kp{i}_x columns are returned.

    Returns:
        np.ndarray: Structured array with fields (tracklet_id, t/time, kp0_y, kp0_x, ...).
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

    time_col = next((name for name in header if name in ("t", "time")), None)
    if time_col is None:
        raise ValueError("No time column found. Expected 't' or 'time'.")

    if "tracklet_id" not in header:
        raise ValueError("No 'tracklet_id' column found.")

    optional_int_cols = {"id", "parent_id"}
    wanted = {"tracklet_id", time_col} | set(kp_col_names) | (optional_int_cols & set(header))
    usecols = [i for i, name in enumerate(header) if name in wanted]
    col_names = [header[i] for i in usecols]

    type_for = {"tracklet_id": "i8", time_col: "i8", "id": "i8", "parent_id": "i8"}
    for name in kp_col_names:
        type_for[name] = "f8"
    dtype = [(name, type_for[name]) for name in col_names]

    return np.genfromtxt(
        csv_file_name,
        delimiter=delimiter,
        skip_header=1,
        names=col_names,
        dtype=dtype,
        encoding="utf-8",
        usecols=usecols,
        comments=None,
    )
