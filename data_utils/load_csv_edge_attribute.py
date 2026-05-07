import numpy as np


def load_csv_edge_attribute(edge_attribute_file_name, attribute_name: str, groups=None):
    """Load edge attribute CSV and optionally filter by one or more groups.

    Supports files with columns ``group id_u id_v attribute`` or
    ``group id_u t_u id_v t_v attribute``.  When ``t_u`` / ``t_v``
    columns are present they are silently dropped.

    Args:
        edge_attribute_file_name (str | Path): Path to the edge attribute file.
        attribute_name (str): Name for the attribute column in the returned array.
        groups (list[str] | None): List of group names to keep.
            If None, all groups are returned.

    Returns:
        tuple[np.ndarray, np.ndarray]:
            - edge_attribute_data: Structured array with columns (id_u, id_v, attribute_name).
            - group_data: 1D array of group names (dtype str).
    """
    # Peek at the header to determine which columns are present.
    with open(edge_attribute_file_name, "r") as f:
        header_cols = f.readline().strip().split()

    has_timestamps = "t_u" in header_cols and "t_v" in header_cols

    if has_timestamps:
        dtype = np.dtype(
            [
                ("group", "U20"),
                ("id_u", "i8"),
                ("t_u", "f8"),
                ("id_v", "i8"),
                ("t_v", "f8"),
                (attribute_name, "f8"),
            ]
        )
    else:
        dtype = np.dtype(
            [
                ("group", "U20"),
                ("id_u", "i8"),
                ("id_v", "i8"),
                (attribute_name, "f8"),
            ]
        )

    edge_attribute_data = np.genfromtxt(
        edge_attribute_file_name,
        delimiter=" ",
        names=True,
        dtype=dtype,
        encoding="utf-8",
        autostrip=True,
    )

    # filter rows for chosen groups
    if groups is not None:
        edge_attribute_data = edge_attribute_data[
            np.isin(edge_attribute_data["group"], groups)
        ]

    # extract group data as string array
    group_data = np.asarray(edge_attribute_data["group"], dtype=str)

    # remove group column from edge_attribute_data
    new_dtype = np.dtype(
        [
            ("id_u", "i8"),
            ("id_v", "i8"),
            (attribute_name, "f8"),
        ]
    )
    edge_attribute_data = np.array(
        [tuple(row[field] for field in new_dtype.names) for row in edge_attribute_data],
        dtype=new_dtype,
    )

    return edge_attribute_data, group_data
