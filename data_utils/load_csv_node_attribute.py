import numpy as np


def load_csv_node_attribute(
    node_attribute_file_name: str,
    attribute_name: str = None,
    attribute_prefix: str = None,
    delimiter=" ",
    sequences=None,
):
    """Load node attribute CSV and optionally filter by one or more sequences.

    The CSV file should have columns: sequence, id, t, followed by attribute columns.

    Two use cases:
        1. Single attribute: provide `attribute_name` (e.g., 'pin') to read that column.
        2. Multiple attributes: provide `attribute_prefix` (e.g., 'emb') to read all
           columns starting with '<prefix>_' (e.g., emb_0, emb_1, ...).

    Args:
        node_attribute_file_name (str | Path): Path to the node attribute file.
        attribute_name (str | None): Exact column name to read (e.g., 'pin').
            Mutually exclusive with attribute_prefix.
        attribute_prefix (str | None): Prefix for columns to read (e.g., 'emb' reads emb_*).
            Mutually exclusive with attribute_name.
        delimiter (str): Field delimiter. Defaults to ' '.
        sequences (list[str] | None): List of sequence names to keep.
            If None, all sequences are returned.

    Returns:
        tuple[np.ndarray, np.ndarray]:
            - node_attribute_data: Structured array with columns (id, t, <attributes>).
            - sequence_data: 1D array of sequence names (dtype str).

    Raises:
        ValueError: If neither or both attribute_name and attribute_prefix are provided.
    """
    if (attribute_name is None) == (attribute_prefix is None):
        raise ValueError("Provide exactly one of attribute_name or attribute_prefix")

    with open(node_attribute_file_name, "r") as f:
        header = f.readline().strip().split()

    if attribute_name is not None:
        # single attribute column
        attr_cols = [attribute_name]
    else:
        # multiple columns with prefix
        attr_cols = [h for h in header if h.startswith(f"{attribute_prefix}_")]

    float_fields = [(name, "f8") for name in attr_cols]
    dtype = np.dtype([("sequence", "U20"), ("id", "i8"), ("t", "i8")] + float_fields)

    node_attribute_data = np.genfromtxt(
        node_attribute_file_name,
        delimiter=delimiter,
        names=True,
        dtype=dtype,
        encoding="utf-8",
        autostrip=True,
    )

    if sequences is not None:
        node_attribute_data = node_attribute_data[
            np.isin(node_attribute_data["sequence"], sequences)
        ]

    # extract sequence data as string array
    sequence_data = np.asarray(node_attribute_data["sequence"], dtype=str)

    # remove sequence column from node_attribute_data
    new_dtype = np.dtype([("id", "i8"), ("t", "i8")] + float_fields)
    node_attribute_data = np.array(
        [tuple(row[field] for field in new_dtype.names) for row in node_attribute_data],
        dtype=new_dtype,
    )

    return node_attribute_data, sequence_data
