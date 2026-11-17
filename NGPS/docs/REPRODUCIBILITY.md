# Reproducibility protocol

## Selection and separation

Freeze `configs/subjects_v1.json`, the normalization, sigma and training config
before running experiments, and train one model per method and sigma. Every
neural method selects `model_final.pth` at the configured final epoch; there is
no early stopping, best-clean score or adaptive schedule. Choosing
hyperparameters by comparing held-out test metrics would break this protocol.

Each epoch also records a noisy-only monitoring MSE: replace a fixed random 5%
of pixels in every interior validation slice and score only those coordinates
against their original noisy values. The mask stream resets to `seed+1` every
epoch, and validation has no shuffle or random crop. This is a noisy diagnostic,
not an unbiased clean-risk estimate and not the selection criterion. Adjacent
target methods are not optimized for this masked validation task. Validation and
test clean arrays are never loaded for training or selection, and the tests
enforce that boundary.

## Run records

`metadata.json` and each checkpoint hold the resolved method and training config,
raw YAML values, seed, method and sigma, the full prepared manifest and its
canonical hash, commit and dirty status, device and environment, epoch, fixed
selection schedule, training diagnostics and the full noisy validation history.
Checkpoints also hold model and optimizer states, and the config and manifest
snapshots are saved separately. Snapshot manifest paths keep their original
meaning, so use the original prepared manifest for evaluation, not the relocated
run snapshot.

NGPS logs mean match SSD per patch pixel, maximum match cost, dynamic and static
active fractions, and empty-mask fractions per image and direction. Match cost is
observable but not used as an unvalidated reliability gate. Batch summaries
average over batches and are not subject-level scores. Losses normalize by
active pixels, so empty masks contribute zero rather than NaN.

Python, NumPy and Torch RNGs are seeded. Crop coordinates come from the training
seed and sample index, the DataLoader has a seeded generator, and PyTorch
deterministic algorithms are required with cuDNN benchmarking disabled. Some GPU
algorithms may reject deterministic operations. We do not promise identical bit
patterns across versions or hardware. There is no resume CLI and no RNG-state
restoration; optimizer state alone is not an exact mid-run resume. Repeat an
experiment with an unchanged environment, config and manifest. Do not compare
results from before and after the global-residual or guide changes as if they
shared an implementation. When the recorded commit is dirty, save the diff:
a commit alone does not identify an uncommitted experiment.

## Evaluation definition

The manifest evaluator requires `test` data, checks prepared data hashes,
checkpoint-manifest hash equality and the checkpoint's trained sigma, and uses
each subject's complete volume. Metrics are computed slice-wise, averaged within
subject, then summarized with equal subject weights and a population SD,
separately for each sigma. The CSV holds subject rows plus a separate aggregate
table, and the JSON includes protocol, provenance and all rows. No pooled-slice
weighting is used.

- PSNR: scikit-image with `data_range=1`; the mean of slice PSNRs, not
  volume-MSE PSNR. An exact match gives infinity, serialized as the strict-JSON
  string `Infinity`; SD is null if any subject PSNR is infinite.
- SSIM: scikit-image defaults with `data_range=1`, window 7 (the largest smaller
  odd window for tiny images). Spatial sizes below 3 are rejected.
- HFEN: relative L2 norm of 2D Gaussian-Laplacian differences, sigma 1.5 pixels,
  denominator `norm(reference LoG)+1e-8`. Near-flat references are
  ill-conditioned.
- `SobelGradientMagnitudeSimilarity`: weighted Sobel magnitude agreement,
  constant `1e-4`, max-gradient weights; flat over flat gives 1. It is not exact
  FSIM because it has no phase-congruency component.
- GMSD: the existing 2x downsampled gradient similarity deviation
  implementation, constant 0.0026, not claimed equivalent to every external
  implementation.

Complete-volume inference wall time covers all slices, transfers and stacking,
with backend synchronization. CUDA and Apple MPS runs synchronize before and
after timing; MPS records current and driver allocation because portable
peak-memory telemetry is unavailable. The measurement excludes model loading,
file I/O, hashing and metrics. There is no warmup, so first-volume startup
overhead stays in the number. NLM times include all slice filters. Hardware,
software and thread count are recorded, and times from tiny synthetic inputs are
not performance claims. A CUDA/T4 budget does not transfer to MPS without a new
pilot.

## Remaining scientific risks

Ten subjects, two of them test subjects, cannot support broad generalization
claims. BrainWeb scale, orientation and converter provenance still need
independent verification. Additive Gaussian noise is not realistic
magnitude-MRI Rician or noncentral-chi noise. Neighboring anatomy is not a
repeated noisy observation of the same signal. Guide masks and target search use
noisy inputs and may introduce selection bias, and Top-K averaging changes target
variance. Masked-pixel training is not a strictly blind-spot architecture at
inference. NLM parameters and neural settings here are fixed executable
defaults, not tuned performance claims. We do not claim exact-FSIM validation,
reference-weight numerical equivalence, a medical-data experiment, multi-seed
uncertainty or clinical validation.
