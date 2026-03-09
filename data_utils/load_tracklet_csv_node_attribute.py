import numpy as np


def load_tracklet_csv_node_attribute(
    node_attribute_file_name: str,
    attribute_name: str = None,
    attribute_prefix: str = None,
    delimiter=" ",
):
    """Load tracklet node attribute CSV.

    The CSV file should have columns: tracklet_id, time, k, followed by attribute columns.
    k is an integer (1–25) indicating which sampled detection within the tracklet each row
    corresponds to.

    Two use cases:
        1. Single attribute: provide `attribute_name` (e.g., 'score') to read that column.
        2. Multiple attributes: provide `attribute_prefix` (e.g., 'emb') to read all
           columns starting with '<prefix>_' (e.g., emb_0, emb_1, ...).

    Args:
        node_attribute_file_name (str | Path): Path to the node attribute file.
        attribute_name (str | None): Exact column name to read.
            Mutually exclusive with attribute_prefix.
        attribute_prefix (str | None): Prefix for columns to read (e.g., 'emb' reads emb_*).
            Mutually exclusive with attribute_name.
        delimiter (str): Field delimiter. Defaults to ' '.

    Returns:
        np.ndarray: Structured array with columns (tracklet_id, time, k, <attributes>).

    Raises:
        ValueError: If neither or both attribute_name and attribute_prefix are provided.
    """
    if (attribute_name is None) == (attribute_prefix is None):
        raise ValueError("Provide exactly one of attribute_name or attribute_prefix")

    with open(node_attribute_file_name, "r") as f:
        header = f.readline().strip().split()

    if attribute_name is not None:
        attr_cols = [attribute_name]
    else:
        attr_cols = [h for h in header if h.startswith(f"{attribute_prefix}_")]

    float_fields = [(name, "f8") for name in attr_cols]
    dtype = np.dtype(
        [("tracklet_id", "i8"), ("time", "i8"), ("k", "i8")] + float_fields
    )

    data = np.genfromtxt(
        node_attribute_file_name,
        delimiter=delimiter,
        names=True,
        dtype=dtype,
        encoding="utf-8",
        autostrip=True,
    )

    return data
