"""ExaCA-centred graph extraction for GrainGNN.

grain_field        Field: the grain-id stack ids[k, y, x] with cell size, layer spacing, periodicity, orientations
exaca_reader       read_vtk (ExaCA GrainID VTK file), exaca_theta (orientation table -> theta_x, theta_z per grain),
                   read_run (an ExaCA run directory: G, R, seed, files)
cupf_reader        read_cupf: cuPF 3D field or cross-section file, used to validate against the legacy extractor
periodic_grid      roll / wrap helpers and single-pass connected components on a grid periodic in x, y, both or neither
slice_cleanup      remove_islands, split_bad_corners: the minimal pixel edits made before extraction
boundary_tracing   boundary segments on corners, junction corners and clusters, traced edges, lens handling
network_types      Junction, Edge, Grain, SliceGraph
slice_graph        build_slice_graph: cleanup -> junctions -> refined positions -> grains -> validation (layer 1)
junction_matching  match_junctions: link the junctions of two consecutive slices (triple, then shared grains + distance)
trajectory         Trajectory: the per-run store in slot form, save/load, smooth_tracks (line fit along z)
stack_tracking     track_stack: layer 1 on every slice, slots, events (eliminate, two_sided, switch, appear) (layer 2)
graingnn_view      dead_schedule, paint_dead, track_view: the stack as GrainGNN can represent it (layer 3)
samples            build_samples, calibrate_span: training samples in the GrainHeterograph layout (layer 3)
dataset            save_samples, load_samples, combine_samples: sample pickles and the combined training dataset
__main__           python -m extract <source> [<source> ...] <out.pkl>: sources to one training dataset
"""
from .grain_field import Field
from .exaca_reader import read_vtk, exaca_theta, read_run
from .network_types import Junction, Edge, Grain, SliceGraph
from .slice_graph import build_slice_graph
from .junction_matching import match_junctions, Matching
from .trajectory import Trajectory, smooth_tracks
from .stack_tracking import track_stack
from .graingnn_view import dead_schedule, paint_dead, track_view, schedule_kinds
from .samples import build_samples, calibrate_span
from .dataset import save_samples, load_samples, combine_samples
