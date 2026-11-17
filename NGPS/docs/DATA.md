# Data contract and preparation

Repository: <https://github.com/levib4231/project-ml>

## Sources, manifests and leakage

`configs/subjects_v1.json` is the versioned source identity manifest. Schema 1
holds a global seed (42) and, for each subject, a `subject_id`, `source`, `split`,
source SHA-256, dtype, shape and observed-range record. Paths are relative to the
manifest that contains them, not to the current working directory. The ten
existing subject filenames are listed without changing their contents, and the
checksums and metadata are verified when the source manifest loads. This pass
does not verify the BrainWeb scale.

Split assignment sorts subject IDs by a SHA-256-derived
`(seed, subject, "split")` rank and reserves a rounded 20% each for validation
and test, which gives 6/2/2 for ten subjects. Freeze the manifest for an
experiment: adding or removing subjects can change assignments and needs a new
manifest version.

A prepared manifest records the schema version, source-manifest hash,
normalization, source and file hashes, shapes, dtypes, ranges, per-sigma seeds
and noisy paths, and the canonical JSON is SHA-256 hashed. Validation rejects
duplicate subject IDs, nonconforming split assignments, reused canonical paths,
symlink aliases and existing hard-link aliases, including aliases within one
subject. Source, normalized clean and each noisy realization must use distinct
paths. Training checks noisy file hashes; evaluation checks noisy and clean file
hashes. None of this can discover the same biological subject under unrelated IDs
or a copy under an unrelated path, so source identity still needs human review.
Training and validation items never receive clean paths.

## Explicit normalization

Preparation accepts NumPy `.npy` numeric arrays shaped `(D,H,W)` with finite
values and at least three slices, and converts them explicitly to `float32`:

- `already-unit`: assert that source values already lie in `[0,1]`; do not
  rescale.
- `fixed-range --lower L --upper U`: apply `(x-L)/(U-L)` using finite,
  predeclared `L<U`, and reject source values outside those bounds. No
  per-volume min/max fit happens here.

Bounds should come from verified acquisition or conversion conventions, or from
a policy set on training data, never from test-set statistics. Choosing
`already-unit` only asserts the numerical range; it does not establish physical
intensity meaning. There is no automatic percentile normalization or clipping.
An existing output directory is rejected to prevent accidental overwrite, and a
failed preparation may leave an incomplete directory that you should discard and
rerun elsewhere.

## Gaussian noise and range validation

Noise is independent additive Gaussian at sigma 0.03, 0.06 and 0.10 in unit-scale
intensity coordinates, generated with NumPy `default_rng` (PCG64). Each seed is
the first four bytes of SHA-256 over canonical JSON `[global_seed, subject_id,
"noise:0.03"]` (or `0.06`, `0.10`). It does not depend on processing order or on
Python's randomized hash. The manifest records the NumPy version and seeds.

Clean references must be finite `float32` in `[0,1]`. Noisy observations are not
restricted to that range, because additive Gaussian noise has unbounded support
and clipping would change the distribution, especially in background. Prepared
noisy extrema are recorded, and runtime loaders validate shape, dtype and
finiteness without casting or clipping. Predictions follow the same unbounded
contract. Metrics use the declared clean data range of 1, not noisy extrema.

Guides use float32 bilateral filtering (the historical `sigma_color=35` becomes
`35/255`) followed by float32 median filtering. No uint8 quantization or clipping
remains. Guides exist only during training.

## Optional source conversion dependencies

The supported preparation command reads `.npy` only; it does not modify
`Data/Clean/` or rerun its conversion scripts. You can do NIfTI/MINC conversion
separately with optional `nibabel`; MINC2 may also need `h5py`, and some MINC
encodings need external MINC tools. None of these are core training
requirements. When converting, record the converter and version, orientation and
axis convention, voxel spacing, intensity scaling and source checksum. This
project does not convert header axis order to `(slice,row,column)`
automatically. The existing conversion's anatomical orientation and scale still
need an audit.

Do not commit prepared arrays, run directories or checkpoints. Obtain and use
source medical datasets under their original access and license terms.
