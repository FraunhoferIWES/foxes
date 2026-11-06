# ADR-0012: Chunk Animation Output

- Status: Accepted

## Context

Engine-owned `_write_ani` hooks bypass the adopted output contract and render
separate worker-local animations. The user requires the standard lifecycle:
collect chunk data, combine it during final result evaluation, and save the
animation. Supplied target data must be reused; farm-only runs need shared grids
and point evaluation bounded by `chunk_size_points`.

## Decision

- Implement `ChunkAnimationOutput` under `foxes.models.outputs`, deriving from
  `foxes.core.Output`; remove engine hooks and the superseded chunk helper.
  Register `"create_gif"` and `"create_mp4"` in `ModelBook.outputs`, plus
  `"chunk_gif_<chunk>"` and `"chunk_mp4_<chunk>"` factories for a single state
  chunk. Constructors retain only configuration; validation and preparation
  belong in `initialize()`, so registering unused outputs performs no
  scientific work.
- Load static grid coordinates as a top-level array in `loaded_data.extra_data`,
  eligible for existing engine shared storage. Prepare scientific dependencies
  before running; capture the active engine point-block limit before entering
  the running state.
- Use provided `tdata` fields/coordinates directly. Otherwise calculate shared
  grid points in blocks no larger than the engine point limit. Do not initialize
  dependencies in workers or mutate source containers to restore downwind order.
- Return numerical chunk payloads from `calculate`, buffer only the output's
  private key in `update_results`, then join target tiles followed by state
  chunks in `finalize_results`. Full state selection is `None`; partial frame
  selections are boolean `(FC.STATE,)` variables in `loaded_data["data_vars"]`,
  sliced by engines and applied to chunk payloads. State positions reference
  input states and retain input order.
- Keep raw constructor inputs in the lifecycle stash while running, restoring
  them when the parent retains a stash. Grids, state-sized selections and frame
  buffers do not remain on model instances.
- Render and save one file per variable only during finalization, on the parent.
  Use the existing rectangular XY flow renderer or XY scatter for arbitrary
  targets, without resampling. Always close figures and propagate write errors.
- Add `Downwind.calc_points(output=...)` alongside the existing physical-field
  selection `outputs=...`. Compose with the standard point collector before
  extracting initialization data; keep returned point datasets unchanged.
  Include supplied models in the full point running/unsetting lifecycle.
- Iterative uses one internal convergence collector and initializes/executes
  requested outputs only in its final evaluation pass. Streamline cases use
  Iterative, not Downwind. Keep scientific caches available through chunk
  output calculation and return caches added by outputs.

## Consequences And Verification

Animation filenames have a variable suffix, not a chunk suffix. GIF uses Pillow;
MP4 requires external ffmpeg. Unknown state/chunk selections fail explicitly.
Point-block sizes bound scientific working sets, not final animation storage.
Multiple outputs retain independent buffers outside model instances.

Synthetic tests cover actual GIF frame counts, numerical agreement with separate
point calculations, alternating turbine ordering, shared coordinate broadcast,
point-block limits, out-of-order target/state chunks, selection errors and figure
cleanup. Iterative tests verify the complete requested-output lifecycle is
final-pass-only, and check streamline grid values against a separate point run.
Farm runs cover single, NumPy, threads, process, multiprocess, local Dask
and local distributed clusters; supplied point runs cover NumPy, threads and
process. MP4 is tested when ffmpeg is available; remote infrastructure remains
outside these tests.

## Alternatives

Rendering in workers or `update_results` was rejected: it produces per-chunk
files and can run Matplotlib concurrently in threads. Recalculating supplied
target data was rejected. Initializing models or storing runtime payloads on
`self` was rejected because it violates the shared model lifecycle.
