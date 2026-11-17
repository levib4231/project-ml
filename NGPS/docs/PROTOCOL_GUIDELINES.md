# Checkpoint 1 and Checkpoint 2 protocol guidelines

These guidelines keep the implementation faithful to the Checkpoint 1 proposal
while meeting the T3 Checkpoint 2 requirements. Treat them as a working contract
for code changes, experiments, reporting and follow-up work.

## 1. Source-of-truth hierarchy

Use these sources in order:

1. The frozen Checkpoint 2 protocol for the completed benchmark.
2. `NGPS/configs/subjects_v1.json` for subject identities and splits.
3. `NGPS/docs/CHECKPOINT_2.md` for execution and evidence rules.
4. This document for the combined Checkpoint 1/2 checklist.
5. The Checkpoint 1 report for the original scientific proposal.
6. The NGPS paper for background and paper-specific comparisons.

The team implementation is a course adaptation. Do not call it a numerical
reproduction of the paper unless a separate paper-replication experiment
establishes that claim.

## 2. Checkpoint 1 scientific contract

### Method

- Train a noisy-only, self-supervised 2D NAFNet denoiser.
- Each forward pass receives one noisy center slice, not a concatenated triplet.
- Adjacent noisy slices provide supervision during training.
- Inference uses only the center noisy slice.
- Clean images may generate noisy observations and score frozen predictions.
  They must not supply targets, guides, masks, thresholds, validation scores,
  early stopping or checkpoint selection.

### NGPS target construction

For each center/neighbor direction:

1. Build a guide with bilateral filtering followed by median filtering.
2. Build a direction-specific mask from the guide difference.
3. For masked pixels, compare `7x7` guide patches in a `15x15` neighboring
   search window.
4. Select the lowest-cost `Top-K`, with `K=4` in the frozen protocol.
5. Retrieve center-pixel values from the original raw noisy neighbor, never from
   the filtered guide.
6. Average the retrieved raw values to form the dynamic target.
7. Use raw same-coordinate neighboring values for static pixels.

Guides choose correspondences; raw noisy neighbors supply targets. Never
substitute guide values for regression targets.

### Loss

The local implementation uses:

- static same-coordinate reconstruction;
- dynamic NGPS retrieved-target reconstruction;
- regional consistency on static regions;
- `rc_weight=0.5` in the frozen protocol.

The local code normalizes regional consistency by active pixels, while paper
Eq. 9 uses an unnormalized sum. Keep that difference documented and do not claim
exact paper-loss equivalence.

Confidence-weighted matching is a proposed improvement, not part of the
completed frozen benchmark. Implement it only as a separately labeled follow-up.

## 3. Data and split contract

Use the agreed subject-level split:

| Split | Subjects |
|---|---|
| Train | 04, 06, 18, 41, 42, 43 |
| Validation | 05, 38 |
| Test | 20, 44 |

Rules:

- Assign subjects to splits before generating noise.
- Keep all slices, patches, adjacent pairs and noise draws within the subject
  split.
- Use normalized `[0,1]` clean references and Gaussian corruption:
  `y = x + epsilon`, `epsilon ~ Normal(0, sigma^2)`.
- Evaluate `sigma = 0.03`, `0.06` and `0.10`.
- Do not clip noisy observations or predictions.
- Reuse identical prepared noisy observations across all methods.
- Verify source hashes, prepared-file hashes and manifest hashes before training
  or evaluation.
- Never swap this Gaussian/BrainWeb protocol for the paper's Rician/IXI protocol
  without labeling it as a separate paper-replication experiment.

## 4. Required methods for Checkpoint 2

The required trained comparison is:

1. NGPS: static adjacent targets plus guide-matched dynamic targets and regional
   consistency.
2. Masked-pixel blind spot: a masked-pixel objective with no adjacent targets.
3. Adjacent: unmasked same-coordinate previous/next noisy targets.

NLM is a classical control and does not count as one of the two trained
comparators. NS-N2N is an optional additional project-local comparator.

For every learned method in a matched experiment, keep these constant:

- subjects and noisy files;
- NAFNet capacity;
- optimizer and learning rate;
- seed policy;
- crop and sampling policy;
- optimizer-step budget;
- validation and checkpoint-selection rule;
- hardware and runtime cap.

Only the supervision objective should differ, unless a change is explicitly
identified as a separate ablation.

## 5. Checkpoint 2 selection and success rules

Before opening held-out clean references:

- Freeze the manifest, normalization, noise levels, method list, configuration,
  budget and selection rule.
- Use a fixed final epoch or a validation rule declared in advance.
- Keep validation noisy-only. It is monitoring, not a clean-risk estimate.
- Complete all required learned runs before final testing.
- Validate every checkpoint, schedule, budget record and noisy test input before
  loading test references.

The frozen success criterion is:

> At every sigma, NGPS mean subject PSNR must meet or exceed the stronger trained
> comparator, `max(blindspot, adjacent)`, with zero tolerance.

Report the criterion, the achieved values, the gap and whether it was met. Do not
relax the criterion after seeing test results.

The original medical benchmark used a reduced shared model and a
one-final-epoch/555-update workload because of the measured GPU budget. A longer
schedule or larger model is a new exploratory experiment, not a quiet
continuation of the frozen result.

## 6. Evaluation and reporting contract

Report:

- per-subject PSNR and SSIM with `data_range=1`;
- normalized HFEN or the declared edge-preservation metric;
- complete-volume inference time at every sigma;
- subject-level means, with subjects weighted equally;
- population standard deviation where meaningful;
- representative noisy, prediction, reference and error images;
- structural failures, mask and matching behavior, and noise-model limitations.

Keep timing scopes separate:

- `inference_seconds`: prediction, transfers and assembly, excluding disk I/O
  and model loading;
- `volume_seconds`: load-to-save timing, including file I/O but excluding model
  loading as the protocol declares.
- On Apple Silicon, use `--device mps` and synchronize MPS before and after
  timing. Record MPS current and driver memory when available. MPS has no
  portable peak-memory counter, so do not report its current allocation as peak
  allocation.
- Do not reuse a CUDA/T4 budget for MPS. Pilot and freeze a new
  hardware-specific workload before reporting runtime.

The project metric `SobelGradientMagnitudeSimilarity` is not FSIM. Do not label it
FSIM.

## 7. Change-control rules

Classify every change before implementing it:

- Protocol-preserving bug fix: corrects behavior without changing the declared
  experiment; rerun affected validation.
- Protocol-preserving implementation change: an equivalent implementation with
  evidence and an updated source hash.
- Ablation: changes one component while holding the rest fixed.
- Exploratory follow-up: changes training length, capacity, loss weights,
  matching, noise or data.
- Paper replication: uses IXI/Rician and paper-like settings, separate from the
  course benchmark.

For any new experiment:

- create a new output directory;
- write a new frozen protocol and config;
- preserve the original checkpoints, metrics, figures and success CSV;
- never overwrite or relabel the completed benchmark;
- choose settings from training and validation data only;
- record the source tree, manifest, configuration, seed, hardware and budget.

The original test subjects have already been scored, so later experiments that
use them are exploratory. Stronger confirmation needs additional untouched
subjects or a new held-out split.

## 8. Recommended NGPS improvement workflow

If the goal is better performance, work in this order:

1. Convergence check: run a longer, matched schedule and record training and
   validation curves.
2. Capacity check: compare the budgeted width-16 model with a larger NAFNet
   configuration.
3. Mechanism diagnostics: inspect dynamic-mask fraction, match costs, selected
   offsets, Top-K spread and the separate static, dynamic and RC losses.
4. Validation ablations: test mask threshold, search window, patch size, `K`,
   guide filtering and regional-consistency weight.
5. Confidence weighting: use a predeclared match-cost gate or continuous weight,
   selected only from training and validation data.
6. Paper replication: once the course adaptation is understood, test IXI/Rician
   with paper-like capacity and training.

Do not tune these choices against the already-open test results.

## 9. Claim checklist

Allowed when the evidence supports it:

- "The NGPS implementation runs end to end."
- "NGPS improves the noisy input on the tested subjects and noise levels."
- "NGPS outperforms the masked-pixel baseline in this experiment."
- "NGPS did not meet the predeclared success criterion against the adjacent
  comparator."
- "The proposed advantage was not demonstrated under the tested budget and
  protocol."

Do not claim without new evidence:

- "NGPS is best overall."
- "NGPS reproduces the paper's performance."
- "Confidence weighting improved the benchmark" before it has been implemented
  and tested.
- "The method is clinically effective."
- "The test result is confirmatory" after using that result to choose later
  settings.

## 10. Final pre-submission checklist

- [ ] Checkpoint 1 proposal and all deviations are documented.
- [ ] NGPS, blindspot and adjacent objectives are implemented and trained.
- [ ] NLM is labeled as a classical control.
- [ ] Subject split and shared noise draws are documented.
- [ ] No clean reference entered training, validation, tuning or selection.
- [ ] Frozen protocol and success criterion are included.
- [ ] Per-subject and aggregate results are included for all sigmas.
- [ ] Timing scopes are labeled correctly.
- [ ] Unmet targets and limitations are reported.
- [ ] README contains setup, data/checkpoint access, commands, hardware, runtime
      and limitations.
- [ ] The submitted code and results match the cited GitHub commit.
- [ ] The report is at most three pages, including figures, tables, references
      and contributions.
- [ ] Slides, execution video, code/results, README and GitHub commit are all in
      the Canvas ZIP.
