import numpy as np


def load_csv_edge_attribute(
    edge_attribute_file_name, attribute_name: str, delimiter=" ", sequences=None
):
    """Load edge attribute CSV and optionally filter by one or more sequences.

    Args:
        edge_attribute_file_name (str | Path): Path to the edge attribute file.
        attribute_name (str): Name for the attribute column in the returned array.
        delimiter (str): Field delimiter. Defaults to ' '.
        sequences (list[str] | None): List of sequence names to keep.
            If None, all sequences are returned.

    Returns:
        tuple[np.ndarray, np.ndarray]:
            - edge_attribute_data: Structured array with columns (id_u, id_v, attribute_name).
            - sequence_data: 1D array of sequence names (dtype str).
    """
    dtype = np.dtype(
        [
            ("sequence", "U20"),
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

    # filter rows for chosen sequences
    if sequences is not None:
        edge_attribute_data = edge_attribute_data[
            np.isin(edge_attribute_data["sequence"], sequences)
        ]

    # extract sequence data as string array
    sequence_data = np.asarray(edge_attribute_data["sequence"], dtype=str)

    # remove sequence column from edge_attribute_data
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

    return edge_attribute_data, sequence_data
