from typing import Dict, Tuple, cast

import numpy as np
import numpy.typing as npt


def load_csv_data(
    csv_file_name: str,
    voxel_size: Dict[str, float] | None = None,
    delimiter: str = " ",
    sequences: list[str] | None = None,
) -> Tuple[
    npt.NDArray[np.float64], npt.NDArray[np.str_], dict[int, int], dict[str, int]
]:
    """
    Load CSV data with scaling based on voxel size and optional sequence filtering.

    Args:
        csv_file_name (str): Path to the CSV file.
        voxel_size (Dict[str, float]): Scaling factors for 'x', 'y', (optionally) 'z'.
        delimiter (str, optional): Delimiter used in the CSV. Defaults to ' '.
        sequences (list[str] | None): Optional list of sequence names to keep.
            If None, all sequences are returned.

    Returns:
        Tuple[np.ndarray, np.ndarray, dict[int, int], dict[str, int]]:
            - Numerical data array with scaled coordinates
            - Sequence column as ndarray[str]
            - Mapping from id to original_id (if available)
            - Reverse mapping from "t_original_id" to id
    """
    if voxel_size is None:
        voxel_size = {"x": 1.0, "y": 1.0}

    # Read header to determine which columns exist
    with open(csv_file_name, "r", encoding="utf-8") as f:
        header_line = f.readline().strip()
        # Handle headers that start with '# ' (common in some CSV formats)
        if header_line.startswith("# "):
            header_line = header_line[2:]
        elif header_line.startswith("#"):
            header_line = header_line[1:]
        header = header_line.split(delimiter)

    # Define expected columns and their types
    expected_cols = [
        ("sequence", "U20"),  # force Unicode string
        ("id", "i4"),
        ("t", "i4"),
        ("y", "f8"),
        ("x", "f8"),
        ("parent_id", "i4"),
        ("original_id", "i4"),
    ]
    if "z" in voxel_size:
        expected_cols.insert(3, ("z", "f8"))

    # Filter to only columns that exist in the file, preserving order
    expected_names = [col[0] for col in expected_cols]
    usecols = [i for i, name in enumerate(header) if name in expected_names]
    dtype = [col for col in expected_cols if col[0] in header]

    # Extract column names from dtype for explicit naming
    col_names = [col[0] for col in dtype]

    data = np.genfromtxt(
        csv_file_name,
        delimiter=delimiter,
        skip_header=1,
        names=col_names,
        dtype=dtype,
        encoding="utf-8",
        usecols=usecols,
        comments=None,  # Disable comment handling to support headers starting with #
    )

    if sequences is not None:
        data = data[np.isin(data["sequence"], sequences)]

    # Extract column names safely
    colnames = (
        cast(tuple[str, ...], data.dtype.names) if data.dtype.names is not None else ()
    )
    has_names = bool(colnames)

    # Apply voxel scaling
    data["x"] *= voxel_size.get("x", 1.0)
    data["y"] *= voxel_size.get("y", 1.0)

    if has_names and "z" in colnames and "z" in voxel_size:
        data["z"] *= voxel_size["z"]

    # Define numeric columns dynamically
    numerical_cols = ["id", "t", "y", "x", "parent_id"]
    if has_names and "z" in colnames:
        numerical_cols.insert(2, "z")  # Insert "z" at correct position

    # Handle missing parent_id column - default to -1 for all rows
    has_parent_id = has_names and "parent_id" in colnames

    # Stack numeric data
    numeric_arrays = []
    for col in numerical_cols:
        if col == "parent_id" and not has_parent_id:
            numeric_arrays.append(np.full(len(data), -1, dtype=np.int32))
        else:
            numeric_arrays.append(data[col])
    numerical_data = np.column_stack(numeric_arrays)
    sequence_data = data["sequence"]

    # Mapping from id to original_id (if exists)
    if has_names and "original_id" in colnames:
        mapping = {
            int(id_): int(orig_id)
            for id_, orig_id in zip(data["id"], data["original_id"])
        }
        reverse_mapping = {
            f"{int(t)}_{int(orig_id)}": int(id_)
            for id_, t, orig_id in zip(data["id"], data["t"], data["original_id"])
        }
    else:
        mapping = {}
        reverse_mapping = {}

    return numerical_data, sequence_data, mapping, reverse_mapping

