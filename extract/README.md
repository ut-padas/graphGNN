# extract: GrainGNN training data from ExaCA

Turns the grain-id field of a solidification simulation into the training samples `train.py` reads.
Input: an ExaCA run (or a cuPF file). Output: one pickle. Self-contained; the training code is not needed.

## Install

```sh
pip install .              # from the repository root; installs the `extract` package only
pip install ".[cupf]"      # also h5py, for cuPF files
```
Python >= 3.9, numpy, scipy, vtk. Without installing: run from the repository root, or put it on `PYTHONPATH`.

## Use

```sh
python -m extract runs/G1.9_V0.558_s1001 runs/G1.9_V0.558_s1002 ... dataset_exaca.pkl \
    --orientations GrainOrientationVectors.csv --smooth 5 --jobs 8
python train.py --dataset exaca ...
```
`graingnn-extract` is the same command after `pip install`.

A source is one of

| source | what is read |
|---|---|
| ExaCA run directory, or its `input.json` | G, R, seed, the VTK file and the orientation csv from the json |
| ExaCA `GrainID` VTK file | `--G` K/um, `--R` m/s, `--seed`; `--orientations` |
| cuPF h5 file (3D `alpha` or QoI `cross_sec`) | G, R, seed from the file name |
| pickle written by this command | included as it is |

Several sources give one dataset, stack order shuffled with a fixed seed (`train.py` takes the tail of the list as
the validation split). A cluster campaign extracts each stack in its own job and merges the pickles with one call.

| option | default | |
|---|---|---|
| `--orientations CSV` | from the run | ExaCA orientation table, `share/ExaCA/GrainOrientationVectors.csv` |
| `--span N` | per stack | layers between a state and its labels; default: the paper's rule, the largest span with fewer than 9 switches and 4.5 eliminations per window |
| `--smooth N` | 0 | window in layers of the line fit along z applied to junction tracks (5 used) |
| `--z-range UM` | 48 | height that z and span are scaled by (the paper's 120 frames of 0.4 um) |
| `--theta-x` | tilted | azimuth of the tilted `<100>` axis, or `lateral` for the in-plane axis |
| `--jobs N` | 1 | stacks extracted in parallel |
| `--shuffle-seed N` | 35 | `-1` keeps the given order |
| `--save-traj DIR` | | also the raw and painted trajectories per stack, npz, for inspection |

## What ExaCA must provide

- Directional solidification, periodic in x and y, final `GrainID` written once (`Printing.Interlayer.Fields: ["GrainID"]`,
  ASCII or binary VTK). No time snapshots are needed.
- `input.json` kept in the run directory: `TemperatureData.G` (K/m) and `R` (K/s) become G = G/1e6 K/um and
  V = R/G m/s; `RandomSeed` is the seed.
- The orientation table used by the run.
- [`examples/input.json`](examples/input.json): the campaign setup, 0.25 um cells, 40 x 40 x 50 um, no nucleation,
  `Substrate.Epsilon` 0.002 (cells per degree; keep `Epsilon x CellSize` fixed when changing the cell size).

ExaCA version notes (pipit branch, 07f2d5d). Three fixes were applied for the campaign: `acos(|z|)` in the capture
rule (CAinterface.hpp), substrate seed coordinates padded to local and the periodic period `nx - 2` (CAcelldata.hpp).
`UndercoolingStart` is written as zero everywhere (`fmin` where `fmax` is meant); it is not used here. Above 82.6 M
cells an int32 overflow aborts the run; use two or more MPI ranks.

## Output

A list of samples. A sample is one layer k with the labels of layer k + span, a plain namespace in the attribute
layout of `graph_datastruct.GrainHeterograph`:

| attribute | content |
|---|---|
| `feature_dicts` | `grain` (G, 10): x, y, z, area, cosx, sinx, cosz, sinz, span, darea history; `joint` (J, 8): x, y, z, G, R, span, dx, dy history |
| `target_dicts` | `grain` darea x20; `joint` dx, dy x5; `grain_event` eliminated; `edge_event` neighbour switch (1), kept (0), gone (-100); `edge` signed length change x2.5 |
| `mask` | `grain` has cells; `joint` same junction with the same grains at k and k + span; `edge` still joined |
| `edge_index_dicts`, `edge_weight_dicts` | grain -> joint and joint -> grain (distance to centroid), joint -> joint (edge length) |
| `features`, `targets`, `targets_scaling`, `edge_type`, `events` | names and scalings; the model sizes itself from them |
| `physical_params` | G, R, seed, height (layer index) |
| `vertex2joint`, `edges`, `gids`, `span` | junction triples, edge list, original grain ids, span in layers |

Units as in the paper: positions in the unit square, area as a fraction of the slice, z and span as fractions of
`--z-range`, G as 1 - G/10, R as R/2, displacements the shortest periodic difference.

## How it works

| stage | modules | |
|---|---|---|
| 1 slice | `slice_cleanup`, `boundary_tracing`, `slice_graph` | one layer -> junctions, boundaries, grains; islands repainted, degree-4 corners split, positions by line fit, Euler check |
| 2 track | `junction_matching`, `stack_tracking`, `trajectory` | junctions matched layer to layer by grain triple; events: eliminate, two-sided, enclosed, switch |
| 3 view | `graingnn_view` | what GrainGNN can represent: a grain is dead from its first layer with at most two junctions, its cells go to the neighbours |
| 3 samples | `samples`, `dataset` | states, labels, masks; span calibration; pickles |
| readers | `exaca_reader`, `cupf_reader`, `grain_field`, `periodic_grid` | VTK, h5, run directory; grids periodic in x, y, both or neither |

## Checks

- Every layer of the campaign stacks is a valid trivalent network (200/200 at 0.25 um; cuPF 121/121, the legacy
  extractor 119/121). About 0.05% of cells are changed by the cleanup.
- Against the legacy cuPF extractor on seed 10020: same junction rows (5622), 99.9% same grain triples, label RMS
  2.42 vs 2.44 px, switching edges 159 vs 158.
- 48 ms per 160 x 160 layer, about 10 s per 200-layer stack.

```sh
pip install ".[test]"
pytest tests/test_slice_cleanup.py tests/test_slice_graph.py tests/test_stack_tracking.py tests/test_samples.py tests/test_cli.py
```

Rollout evaluation against ExaCA, which needs the training code: `exaca_rollout_data.py` at the repository root.
