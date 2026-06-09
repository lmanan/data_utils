import numpy as np

# Columns that index a row rather than carry an attribute value. The embeddings
# CSV may be keyed by detection `id` (current) or by `tracklet_id` (legacy);
# `k` (sampled-detection index) and the time column are optional. Whichever of
# these are present are loaded so the caller can join on the one it needs.
META_COLS = ("tracklet_id", "id", "time", "t", "k")


def load_tracklet_csv_node_attribute(
    node_attribute_file_name: str,
    attribute_name: str = None,
    attribute_prefix: str = None,
    delimiter=" ",
):
    """Load tracklet node attribute CSV.

    Recognised index columns (any subset, in file order): tracklet_id, id,
    time/t, k. At least one of `id` or `tracklet_id` is expected so the caller
    can associate each row with a tracklet. These are followed by attribute
    columns.

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
        np.ndarray: Structured array whose leading fields are whichever of
        META_COLS are present, followed by the requested attribute columns.

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

    meta_cols = [h for h in header if h in META_COLS]
    col_names = meta_cols + attr_cols
    usecols = [header.index(name) for name in col_names]

    int_fields = [(name, "i8") for name in meta_cols]
    float_fields = [(name, "f8") for name in attr_cols]
    dtype = np.dtype(int_fields + float_fields)

    data = np.genfromtxt(
        node_attribute_file_name,
        delimiter=delimiter,
        skip_header=1,
        names=col_names,
        dtype=dtype,
        usecols=usecols,
        encoding="utf-8",
        autostrip=True,
        comments=None,
    )

    return data
