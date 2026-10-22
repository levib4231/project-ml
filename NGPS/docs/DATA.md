# Data contract and preparation

Repository: https://github.com/levib4231/project-ml

## Sources, manifests and leakage

`configs/subjects_v1.json` is the versioned source identity manifest. Schema 1
contains a global seed (42) and one `subject_id`, `source`, `split`, source
SHA-256, dtype, shape, and observed-range record per subject. Paths are relative
to the containing manifest, not the current working directory. The ten existing
subject filenames are listed without modifying their contents; recorded checksums
and metadata are verified when the source manifest is loaded. **The BrainWeb
scale is not verified by this pass.**

Split assignment sorts subject IDs by SHA-256-derived `(seed, subject, "split")`
rank and reserves rounded 20% each for validation and test. Ten subjects give
6/2/2. Freeze the manifest for an experiment: adding/removing subjects can change
assignments and requires a new manifest version. A prepared manifest records
schema version, source-manifest hash, normalization, source/file hashes, shapes,
dtypes, ranges, per-sigma seeds, and noisy paths. Canonical JSON is SHA-256 hashed.

Validation rejects duplicate subject IDs, nonconforming split assignments,
reused canonical paths, symlink aliases and existing hard-link aliases (including
within a subject). Paths for source, normalized clean and each noisy realization
must be distinct. Training checks noisy file hashes; evaluation checks noisy and
clean file hashes. These checks cannot discover the same biological subject
under unrelated IDs or copies under unrelated paths; source identity provenance
still needs human review. Training/validation items never receive clean paths.

## Explicit normalization

Preparation accepts NumPy `.npy` numeric arrays with `(D,H,W)`, finite values,
and at least three slices. It explicitly converts to `float32`:

- `already-unit`: assert source values already lie in `[0,1]`; do not rescale.
- `fixed-range --lower L --upper U`: apply `(x-L)/(U-L)` using finite predeclared
  `L<U`; reject source values outside those bounds. No per-volume min/max fit.

Bounds should come from verified acquisition/conversion conventions or a policy
established on training data, not test-set statistics. Choosing `already-unit`
only asserts numerical range; it does not establish physical intensity meaning.
No automatic percentile normalization or clipping is implemented. An existing
output directory is rejected to prevent accidental overwrite. Failed preparation
may leave an incomplete directory; discard it and rerun to a fresh location.

## Gaussian noise and range validation

Noise is independent additive Gaussian at sigma **0.03, 0.06, 0.10 in unit-scale
intensity coordinates**, generated with NumPy `default_rng` (PCG64). Each seed is
the first four bytes of SHA-256 over canonical JSON `[global_seed, subject_id,
"noise:0.03"]` (or `0.06`, `0.10`). It does not depend on processing order or
Python's randomized hash. The NumPy version and seeds are recorded.

Clean references must be finite `float32` in `[0,1]`. **Noisy observations are
not restricted to `[0,1]`**: additive Gaussian noise has unbounded support.
Clipping would change the noise distribution, particularly in background.
Prepared noisy extrema are recorded; runtime loaders validate shape, dtype and
finiteness without casting or clipping. Predictions use the same unbounded
contract. Metrics use the declared clean data range of 1, not noisy extrema.

Guides use float32 bilateral filtering (the historical `sigma_color=35` is
converted to `35/255`), followed by float32 median filtering. No uint8
quantization or clipping remains. Guides are only training-time constructions.

## Optional source conversion dependencies

The supported preparation command consumes `.npy` only; it does not modify
`Data/Clean/` or re-run its conversion scripts. NIfTI/MINC conversion can be done
separately with optional `nibabel`; MINC2 support may also require `h5py`, and
some MINC encodings require external MINC tools. These are not core training
requirements. Record converter/version, orientation/axis convention, voxel
spacing, intensity scaling and source checksum when converting. Header axis
order is not automatically converted to `(slice,row,column)` by this project.
The existing conversion's anatomical orientation and scale still require audit.

Do not commit prepared arrays, run directories or checkpoints. Obtain and use
source medical datasets under their original access and license terms.
