# data_utils

Utilities for turning raw acquisitions (image directories, movies) into zarr
containers that the rest of the stack — training datasets, the visualizer, the
tracker — can consume without per-dataset special-casing.

## Zarr conventions

Every zarr this repo writes should follow the same shape. `create_zarr.py` is
the reference implementation; new converters must match it.

### Layout

```
container.zarr/
└── <group>/          # one group per movie / acquisition / sequence
    ├── img           # (C, T, [Z], Y, X)
    └── mask          # (1, T, [Z], Y, X), uint32, optional
```

- Axes are always channel-first and time-second: `(C, T, Y, X)` in 2D,
  `(C, T, Z, Y, X)` in 3D. Even a single-channel movie keeps its `C` axis.
- `img` keeps the dtype of the source data. Do not silently upcast or rescale.
- `mask` is `uint32` and single-channel — label images, not one-hot.

### Compression

Blosc, zstd, bitshuffle. This is the default in `create_zarr.py`:

```python
DEFAULT_COMPRESSOR = Blosc(cname="zstd", clevel=7, shuffle=Blosc.BITSHUFFLE)
```

Use Blosc rather than an image codec (per-frame JPEG/PNG): it is lossless and
TensorStore-compatible, so tile serving stays fast and does not require a Python
process to decode chunks. Lower `clevel` if write throughput matters more than
size on a given dataset; keep zstd + bitshuffle.

### Chunking

Chunks are `(1, chunk_t, *spatial_chunks)` — one channel per chunk, clipped to
the array extent. Defaults: `(64, 256, 256)` spatially in 3D, `(256, 256)` in 2D.

`chunk_t` is 1 for frame-at-a-time sources. For long movies of small frames, a
larger `chunk_t` (e.g. 32) is much better — one chunk per frame across 100k
frames produces a pathological number of tiny files. When `chunk_t > 1`, buffer
frames and flush a whole chunk at once; writing frame-by-frame into a multi-frame
chunk forces a read-modify-write per frame.

### `.zattrs` schema

Attributes are written on `img` (and mirrored onto `mask`), describing the
**non-channel** axes in order — `(t, [z], y, x)`:

| key | meaning |
|---|---|
| `axis_names` | full axis list *including* channel, e.g. `["c","t","z","y","x"]` |
| `resolution` | size per non-channel axis: time first, then spatial |
| `offset` | origin per non-channel axis, same units as `resolution` |
| `units` | unit per non-channel axis, e.g. `["s","nm","nm","nm"]` |
| `voxel_size_um` | spatial voxel size in µm, for humans |
| `time_resolution_s` | seconds between frames |

`resolution` is deliberately **mixed-unit**: seconds on the time axis,
nanometers on the spatial axes (i.e. `voxel_size_um * 1000`). `units` is what
declares this per axis, and consumers read it that way. A 3D example:

```json
{
    "axis_names": ["c", "t", "z", "y", "x"],
    "resolution": [1.0, 2031.0, 406.0, 406.0],
    "offset": [0, 0, 0, 0],
    "units": ["s", "nm", "nm", "nm"],
    "voxel_size_um": [2.031, 0.406, 0.406],
    "time_resolution_s": 1.0
}
```

### When the physical scale is unknown

**Do not invent one.** If no voxel size is supplied, write only:

```json
{
    "axis_names": ["c", "t", "y", "x"],
    "resolution": [1.0, 1.0, 1.0],
    "offset": [0, 0, 0]
}
```

Omitting `units` is meaningful, not lazy: `visualizer.read_axis_metadata` treats
a dataset with no `units` as dimensionless voxel/frame indices, so the image and
the (also dimensionless) annotation layers stay aligned. Defaulting to a fake
1 µm voxel would instead assert a physical scale that is not true and would
misscale overlays. Physical units are opt-in, via an explicit voxel size.

Note also that neuroglancer only accepts SI-based units; a non-SI label such as
`"px"` is silently dropped to dimensionless, so it buys nothing over omitting
the key.

### Downsampling

If frames are downsampled on the way in, the *stored* pixel size grows: record
`voxel_size_um = source_pixel_size_um * downsample`. The zarr should describe
what it actually contains, not the source acquisition.

## Conventions

- Type hints on all function signatures.
- No comments unless the reason is non-obvious; no docstrings on trivial helpers.
- Anything that changes on-disk metadata: check who reads it first
  (`visualizer`, trackers, training datasets) before renaming or dropping a key.
