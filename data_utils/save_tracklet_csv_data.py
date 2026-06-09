import logging
from pathlib import Path
from typing import Iterable

import networkx as nx
import numpy as np

from data_utils.load_tracklet_csv_data import load_tracklet_csv_data

logger = logging.getLogger(__name__)


def save_tracklet_csv_data(
    solution_graph,
    csv_path: str,
    out_path: str | Path,
    group: str = "group",
    use_original_tracklet_id: bool = False,
) -> None:
    """
    Export a detection-level centroid CSV from a solution graph.

    The output file has columns: group id t y x parent_id tracklet_id

    Centroid coordinates are computed as the mean over all keypoint (kp*_y,
    kp*_x) columns in the tracklet CSV.  Detections belonging to the same
    chain are ordered by their position in the chain (not purely by time) so
    that backward edges are handled correctly, and parent_id links each
    detection to the immediately preceding detection within its chain.

    Args:
        solution_graph: Directed networkx graph whose nodes are tracklet IDs
            and whose edges represent links between consecutive tracklets.
        csv_path: Path to the tracklet CSV file (space-delimited, with header).
        out_path: Destination path for the output CSV file.
        group: Sequence name written into the ``group`` column.
        use_original_tracklet_id: If True, the tracklet_id column contains the
            original tracklet_id from the input CSV (one per detection). If
            False (default), it contains the chain/cluster id shared by all
            detections in the same stitched chain.
    """
    # Walk chains; assign each tracklet its (chain_id, order_in_chain).
    # Sorting by (chain_id, order_in_chain, t) rather than just (chain_id, t)
    # is necessary when backward edges create tracklets that overlap in time:
    # sorting by t alone would interleave detections from two consecutive
    # tracklets, breaking the parent_id linkage.
    parent_lookup = {dst: src for src, dst in solution_graph.edges()}
    roots = sorted(n for n in solution_graph.nodes() if n not in parent_lookup)
    tracklet_chain_id: dict = {}
    tracklet_order: dict = {}
    for chain_id, root in enumerate(roots, start=1):
        node, order = root, 0
        while node is not None:
            tracklet_chain_id[node] = chain_id
            tracklet_order[node] = order
            order += 1
            children = list(solution_graph.successors(node))
            node = children[0] if children else None

    tracklet_data = load_tracklet_csv_data(csv_path)
    kp_y_fields = [
        f for f in tracklet_data.dtype.names if f.startswith("kp") and f.endswith("_y")
    ]
    kp_x_fields = [
        f for f in tracklet_data.dtype.names if f.startswith("kp") and f.endswith("_x")
    ]

    sel_mask = np.isin(tracklet_data["tracklet_id"], list(solution_graph.nodes()))
    det = tracklet_data[sel_mask]

    time_col = next(name for name in tracklet_data.dtype.names if name in ("time", "t"))
    original_tracklet_ids = np.array([int(tid) for tid in det["tracklet_id"]])
    chain_ids = np.array([tracklet_chain_id[int(tid)] for tid in det["tracklet_id"]])
    orders = np.array([tracklet_order[int(tid)] for tid in det["tracklet_id"]])
    times = det[time_col]
    ys = np.stack([det[f] for f in kp_y_fields], axis=1).mean(axis=1)
    xs = np.stack([det[f] for f in kp_x_fields], axis=1).mean(axis=1)

    sort_idx = np.lexsort((times, orders, chain_ids))
    original_tracklet_ids = original_tracklet_ids[sort_idx]
    chain_ids = chain_ids[sort_idx]
    times = times[sort_idx]
    ys = ys[sort_idx]
    xs = xs[sort_idx]

    n = len(times)
    ids = np.arange(1, n + 1)
    parent_ids = np.zeros(n, dtype=np.int64)
    prev_id: dict = {}
    for i in range(n):
        c = int(chain_ids[i])
        if c in prev_id:
            parent_ids[i] = prev_id[c]
        prev_id[c] = ids[i]

    tracklet_id_col = original_tracklet_ids if use_original_tracklet_id else chain_ids

    out_path = Path(out_path)
    out_path.parent.mkdir(exist_ok=True, parents=True)
    with open(out_path, "w") as f:
        f.write("group id t y x parent_id tracklet_id\n")
        for i in range(n):
            f.write(
                f"{group} {ids[i]} {times[i]}"
                f" {ys[i]:.4f} {xs[i]:.4f} {parent_ids[i]} {tracklet_id_col[i]}\n"
            )
    logger.info("Exported %s (%d detections, %d chains)", out_path, n, len(roots))


def save_solution_keypoints_csv(
    solution_graph,
    csv_path: str | Path,
    out_path: str | Path,
    group: str,
) -> None:
    """Export a keypoints CSV from a solution graph.

    Mirrors the column layout of save_tracklet_csv_data (group id t y x
    parent_id tracklet_id) with all kp* columns appended, so that the id
    column maps one-to-one between this file and solution_centroids.csv.

    Args:
        solution_graph: Directed networkx graph whose nodes are tracklet IDs.
        csv_path: Path to the source tracklet keypoints CSV.
        out_path: Destination path for the output CSV.
        group: Sequence name written into the ``group`` column.
    """
    parent_lookup = {dst: src for src, dst in solution_graph.edges()}
    roots = sorted(n for n in solution_graph.nodes() if n not in parent_lookup)
    tracklet_chain_id: dict = {}
    tracklet_order: dict = {}
    for chain_id, root in enumerate(roots, start=1):
        node, order = root, 0
        while node is not None:
            tracklet_chain_id[node] = chain_id
            tracklet_order[node] = order
            order += 1
            children = list(solution_graph.successors(node))
            node = children[0] if children else None

    tracklet_data = load_tracklet_csv_data(csv_path)
    kp_y_fields = [f for f in tracklet_data.dtype.names if f.startswith("kp") and f.endswith("_y")]
    kp_x_fields = [f for f in tracklet_data.dtype.names if f.startswith("kp") and f.endswith("_x")]
    kp_fields = [f for f in tracklet_data.dtype.names if f.startswith("kp")]

    sel_mask = np.isin(tracklet_data["tracklet_id"], list(solution_graph.nodes()))
    det = tracklet_data[sel_mask]

    time_col = next(name for name in tracklet_data.dtype.names if name in ("time", "t"))
    chain_ids = np.array([tracklet_chain_id[int(tid)] for tid in det["tracklet_id"]])
    orders = np.array([tracklet_order[int(tid)] for tid in det["tracklet_id"]])
    times = det[time_col]
    ys = np.stack([det[f] for f in kp_y_fields], axis=1).mean(axis=1)
    xs = np.stack([det[f] for f in kp_x_fields], axis=1).mean(axis=1)
    kp_values = {f: det[f] for f in kp_fields}
    original_ids = det["id"]
    original_tracklet_ids = det["tracklet_id"]

    sort_idx = np.lexsort((times, orders, chain_ids))
    chain_ids = chain_ids[sort_idx]
    times = times[sort_idx]
    ys = ys[sort_idx]
    xs = xs[sort_idx]
    kp_values = {f: v[sort_idx] for f, v in kp_values.items()}
    original_ids = original_ids[sort_idx]
    original_tracklet_ids = original_tracklet_ids[sort_idx]

    n = len(times)
    ids = np.arange(1, n + 1)
    parent_ids = np.zeros(n, dtype=np.int64)
    prev_id: dict = {}
    for i in range(n):
        c = int(chain_ids[i])
        if c in prev_id:
            parent_ids[i] = prev_id[c]
        prev_id[c] = ids[i]

    out_path = Path(out_path)
    out_path.parent.mkdir(exist_ok=True, parents=True)
    with open(out_path, "w") as f:
        f.write("group id t y x parent_id tracklet_id original_id original_tracklet_id " + " ".join(kp_fields) + "\n")
        for i in range(n):
            kp_str = " ".join(str(kp_values[field][i]) for field in kp_fields)
            f.write(
                f"{group} {ids[i]} {times[i]}"
                f" {ys[i]:.4f} {xs[i]:.4f} {parent_ids[i]} {chain_ids[i]}"
                f" {original_ids[i]} {original_tracklet_ids[i]} {kp_str}\n"
            )
    logger.info("Exported %s (%d detections, %d chains)", out_path, n, len(roots))


def save_raw_tracklets_csv(
    tracklet_ids: Iterable[int],
    csv_path: str | Path,
    out_path: str | Path,
    group: str = "group",
) -> None:
    """Save a centroid CSV for a set of tracklets, each treated as an independent chain.

    Reuses save_tracklet_csv_data with a graph of isolated nodes (no edges),
    so detections within each tracklet are linked sequentially by time.

    Args:
        tracklet_ids: Tracklet IDs to include.
        csv_path: Path to the source tracklet keypoints CSV.
        out_path: Destination path for the output CSV.
        group: Value written into the 'group' column.
    """
    G = nx.DiGraph()
    for tid in tracklet_ids:
        G.add_node(int(tid))
    save_tracklet_csv_data(G, csv_path, out_path, group=group, use_original_tracklet_id=True)
