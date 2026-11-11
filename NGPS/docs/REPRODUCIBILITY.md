# Reproducibility protocol

## Selection and separation

Freeze `configs/subjects_v1.json`, normalization, sigma and training config
**before** running experiments. Train one model per method/sigma. All neural
methods select `model_final.pth` at the configured final epoch; no early stopping,
best-clean score or adaptive schedule is implemented. Comparing held-out test
metrics to choose hyperparameters would violate this protocol.

Each epoch also records a noisy-only monitoring MSE: replace a fixed random 5%
of pixels in every interior validation slice and score only those coordinates
against their original noisy values. The mask stream is reset to `seed+1` every
epoch; validation has no shuffle or random crop. This is a noisy diagnostic,
not an unbiased clean-risk estimate or the selection criterion. Adjacent-target
methods are not optimized for this masked validation task. Validation/test clean
arrays are never loaded for training or selection; tests enforce that boundary.

## Run records

`metadata.json` and each checkpoint contain resolved method/training config,
raw YAML values, seed, method/sigma, full prepared manifest and canonical hash,
commit/dirty status, device/environment, epoch, fixed selection schedule,
training diagnostics and full noisy validation history. Checkpoints also hold
model and optimizer states. Config and manifest snapshots are saved separately.
Snapshot manifest paths retain their original meaning; use the original prepared
manifest for evaluation, not the relocated run snapshot.

NGPS logs mean match SSD per patch pixel, maximum match cost and dynamic/static
active fractions plus empty-mask fractions per image/direction. Match cost is
observable, not used as an unvalidated reliability gate. Batch summaries are
averaged by batch (not interpreted as subject-level scores). Losses normalize
by active pixels; empty masks contribute zero, not NaN.

Python, NumPy and Torch RNGs are seeded; crop coordinates are derived from the
training seed and sample index; DataLoader has a seeded generator;
PyTorch deterministic algorithms are required and cuDNN benchmarking disabled.
GPU algorithms may reject unsupported deterministic operations. Identical
cross-version/hardware bit patterns are not promised. There is no resume CLI or
RNG-state restoration; optimizer state alone is not an exact mid-run resume.
Run an unchanged environment/config/manifest for repeat experiments. Do not
compare results from before and after the global-residual/guide changes as if
they used the same implementation. Save the diff when the recorded commit is
dirty; commit alone does not identify an uncommitted experiment.

## Evaluation definition

The manifest evaluator enforces `test`, prepared data hashes, checkpoint-manifest
hash equality and the checkpoint's trained sigma. Each subject uses its complete
volume. Metrics are computed slice-wise then averaged within subject; aggregate
means and population SD give each subject equal weight, separately by sigma.
CSV contains both subject rows and a separate aggregate table; JSON includes
protocol/provenance and all rows. No pooled-slice weighting is used.

- PSNR: scikit-image, `data_range=1`; mean of slice PSNRs, not volume-MSE PSNR.
  Exact matches yield infinity, serialized as the strict-JSON string `Infinity`;
  SD is null if any subject PSNR is infinite.
- SSIM: scikit-image defaults with `data_range=1`, window 7 (largest smaller odd
  window for tiny images); spatial sizes below 3 are rejected.
- HFEN: relative L2 norm of 2D Gaussian-Laplacian differences, sigma 1.5 pixels,
  denominator `norm(reference LoG)+1e-8`. Near-flat references are ill-conditioned.
- `SobelGradientMagnitudeSimilarity`: weighted Sobel magnitude agreement,
  constant `1e-4`, max-gradient weights; flat/flat gives 1. **Not exact FSIM**:
  it has no phase-congruency component.
- GMSD: existing 2x downsampled gradient similarity deviation implementation,
  constant 0.0026; not claimed equivalent to every external implementation.

Complete-volume inference wall time includes all slices, transfers and stacking,
with CUDA synchronization. It excludes model loading, file I/O, hashing and
metrics. There is no warmup; first-volume startup overhead remains. NLM times
include all slice filters. Hardware/software and thread count are recorded;
times from tiny synthetic inputs are not performance claims.

## Remaining scientific risks

Ten subjects (two test subjects) cannot support broad generalization claims.
BrainWeb scale/orientation/converter provenance need independent verification.
Additive Gaussian noise is not realistic magnitude-MRI Rician/noncentral-chi
noise. Neighboring anatomy is not a repeated noisy observation of exactly the
same signal. Guide masks and target search use noisy inputs and may introduce
selection bias; Top-K averaging changes target variance. Masked-pixel training
is not a strictly blind-spot architecture at inference. NLM parameters and
neural settings here are fixed executable defaults, not tuned performance
claims. No exact-FSIM validation, reference-weight numerical equivalence,
medical-data experiment, multi-seed uncertainty or clinical validation is claimed.
