import logging
from argparse import ArgumentParser
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import yaml
from skimage.io import imread
from skimage.measure import regionprops

from data_utils.image_utils import get_image_files

# ----------------- Logging Setup -----------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)
# -------------------------------------------------


def create_csv(
    group_names: List[str],
    detections_csv_file_name: str,
    mask_dir_names: Optional[List[str]] = None,
    zarr_container: Optional[str] = None,
    man_track_file_names: Optional[List[str]] = None,
    unique_ids: bool = True,
) -> None:
    """
    Create CSV files from segmentation masks and manual track files.

    This function creates a CSV from available segmentation masks and a corresponding
    `man_track.txt`.

    The CSV file provides an ID for each segmentation, with 7 or 8 columns:
    `group id time [z] y x parent_id original_id`.

    Masks can be provided either as image directories (via `mask_dir_names`) or
    as a zarr container (via `zarr_container`). Exactly one must be specified.

    Parameters
    ----------
    group_names : List[str]
        List of group names corresponding to each mask directory.
        When using `zarr_container`, these are the group names within the container.
    man_track_file_names : List[str] | None
        TXT files with CTC-style 4 columns (track_id time_start time_end parent_track_id).
        If not provided, parent id will be set to -1.
    detections_csv_file_name : str
        Output CSV filename with segmentation IDs and parent IDs.
        Columns are [group id t [z] y x parent_id original_id].
    mask_dir_names : List[str] | None
        List of directories containing segmentation masks.
        Supported formats: tif, tiff, png, jpg, jpeg.
    zarr_container : str | None
        Path to a zarr container. Each `group_name` is a group within the
        container, and each group must contain a dataset called 'mask'.
    unique_ids : bool
        If True, assigns unique IDs across the entire dataset (starting from 1).
        If False, preserves the original label IDs from the masks.

    """

    assert (mask_dir_names is None) != (
        zarr_container is None
    ), "Exactly one of `mask_dir_names` or `zarr_container` must be provided."

    if zarr_container is not None:
        import zarr

        store = zarr.open(zarr_container, mode="r")

    all_data: List[List[Union[str, int, float]]] = []
    for seq_index in range(len(group_names)):
        group_name = group_names[seq_index]

        if zarr_container is not None:
            mask_array = store[group_name]["mask"]
            num_frames = mask_array.shape[1]
        else:
            mask_dir_name = mask_dir_names[seq_index]
            mask_file_names = get_image_files(Path(mask_dir_name))
            num_frames = len(mask_file_names)
        track_data: Optional[np.ndarray] = None

        if man_track_file_names is not None:
            man_track_file = man_track_file_names[seq_index]
            if Path(man_track_file).exists():
                track_data = np.loadtxt(man_track_file, delimiter=" ")
                logger.info(f"Loaded manual track file: {man_track_file}")
            else:
                logger.warning(f"Manual track file not found: {man_track_file}")
        else:
            logger.info("No manual track file provided; parent IDs will be set to -1.")

        segmentation_id = 1
        mapping: Dict[int, Dict[int, List[float]]] = {}
        reverse_mapping: Dict[int, Dict[int, int]] = {}
        daughter_parent_mapping: Dict[int, Tuple[List[int], int]] = {}

        logger.info("Processing detections...")
        for time in range(num_frames):
            mapping[time] = {}
            reverse_mapping[time] = {}
            if zarr_container is not None:
                mask = np.array(mask_array[0, time])
            else:
                mask = imread(mask_file_names[time])
            detections = regionprops(mask)
            for detection in detections:
                if unique_ids:
                    assigned_id = segmentation_id
                    segmentation_id += 1
                else:
                    assigned_id = detection.label
                mapping[time][detection.label] = [assigned_id, *detection.centroid]
                reverse_mapping[time][assigned_id] = detection.label

        logger.info("Processing manual track TXT...")
        if track_data is not None:
            for row in track_data:
                id_, time_start, time_end, parent_id = row.astype(int)
                daughter_parent_mapping[id_] = ([], parent_id)
                for time in range(time_start, time_end + 1):
                    if zarr_container is not None:
                        mask = np.array(mask_array[0, time])
                    else:
                        mask = imread(mask_file_names[time])
                    if np.any(mask == id_):
                        daughter_parent_mapping[id_][0].append(time)

        logger.info("Generating CSV output...")
        for time in range(num_frames):
            for key, value in mapping[time].items():
                if key in daughter_parent_mapping:
                    times, parent_track_id = daughter_parent_mapping[key]
                    index = times.index(time)
                    if index == 0:
                        if parent_track_id in daughter_parent_mapping:
                            time_parent = daughter_parent_mapping[parent_track_id][0][
                                -1
                            ]
                            parent_id_updated = mapping[time_parent][parent_track_id][0]
                        else:
                            parent_id_updated = 0
                    else:
                        time_parent = times[index - 1]
                        parent_id_updated = mapping[time_parent][key][0]
                    all_data.append(
                        [
                            group_name,
                            value[0],
                            time,
                            *value[1:],
                            parent_id_updated,
                            reverse_mapping[time][int(value[0])],
                        ]
                    )
                else:
                    all_data.append(
                        [
                            group_name,
                            value[0],
                            time,
                            *value[1:],
                            -1,
                            reverse_mapping[time][int(value[0])],
                        ]
                    )

    header_2d = "group id t y x parent_id original_id"
    header_3d = "group id t z y x parent_id original_id"

    logger.info(f"Saving output CSVs to: {detections_csv_file_name}")
    if len(mask.shape) == 2:
        np.savetxt(
            detections_csv_file_name,
            np.array(all_data, dtype=object),
            delimiter=" ",
            header=header_2d,
            fmt=["%s", "%i", "%i", "%.3f", "%.3f", "%i", "%i"],
        )
    elif len(mask.shape) == 3:
        np.savetxt(
            detections_csv_file_name,
            np.array(all_data, dtype=object),
            delimiter=" ",
            header=header_3d,
            fmt=["%s", "%i", "%i", "%.3f", "%.3f", "%.3f", "%i", "%i"],
        )

    logger.info("CSV creation complete.")


# -------------------- Main CLI ------------------------


def main():
    parser = ArgumentParser(description="Create CSV from YAML config file")
    parser.add_argument(
        "--yaml_config_file_name", required=True, help="Path to YAML config file"
    )
    args = parser.parse_args()

    with open(args.yaml_config_file_name, "r") as f:
        config = yaml.safe_load(f)

    create_csv(
        group_names=config.get("group_names"),
        man_track_file_names=config.get("man_track_file_names"),
        detections_csv_file_name=config["detections_csv_file_name"],
        mask_dir_names=config.get("mask_dir_names"),
        zarr_container=config.get("zarr_container"),
        unique_ids=config.get("unique_ids", True),
    )


if __name__ == "__main__":
    main()
