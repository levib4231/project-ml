# Checkpoint 2 implementation and experiment workflow

This checkout imports runtime code from upstream
`c1ab8bcb2dfa80926c7846048b42356546257f8d` as a source snapshot. The two
histories had diverged, so this is not a fast-forward merge. Local README edits
and related documents are preserved. Historical inventories and splits live in
`NGPS/manifests/archive`; new runs should use the agreed source manifest.

## Submission hierarchy

The completed frozen Gaussian/T4 benchmark is primary and its success criterion
was unmet at every sigma. Later MPS, Rician and loss-component studies are
exploratory; do not substitute their favorable subsets for the primary finding.
See [Submission evidence](SUBMISSION_EVIDENCE.md) for packaged records, access,
source-version boundaries, failure/recovery notes and numerical verification.

## Current evidence

- We downloaded all ten original compressed MINC files from BrainWeb on
  November 11, 2026. Their decompressed contents match the local files exactly.
- The audited MINC real-intensity conversion reproduces all ten historical NumPy
  arrays exactly. Receipt: `NGPS/manifests/brainweb_verification.json`.
- Normalization keeps the course fixed scale: MINC real intensities become
  float32, are clipped to `[0,4095]` and divided by 4095. The MINC stored integer
  range is not the real intensity range. These arrays fill only part of `[0,1]`,
  so this is not per-subject min/max normalization.
- Source history names `pn3_rf20` coil settings. These are public fixed
  reference signals, not established noise-free anatomy. The experiment measures
  recovery after extra Gaussian corruption and cannot show clinical
  effectiveness.
- BrainWeb asks for attribution, and its public pages do not grant an explicit
  redistribution license. Download source volumes directly, leave them out of
  code exports, and do not apply the code's MIT license to the data.

Sources: [20 normal models](https://brainweb.bic.mni.mcgill.ca/brainweb/anatomic_normal_20.html),
[subject 04 download information](https://brainweb.bic.mni.mcgill.ca/cgi/brainweb1?alias=subject04_t1w&download=1),
[MINC/raw format explanation](https://brainweb.bic.mni.mcgill.ca/brainweb/about_data_formats.html),
[requested references](https://brainweb.bic.mni.mcgill.ca/brainweb/#references).

## Environment and hosted execution

Use Python 3.11 or 3.12 in an isolated environment. Install the pinned hosted
runtime with `python -m pip install -r NGPS/requirements-colab.txt` before
importing PyTorch. The general requirements still work for CPU checks. Take the
actual versions and hardware from the generated run metadata.

Open `NGPS/notebooks/checkpoint2_colab.ipynb` in Colab and select a T4 with
Runtime version 2026.07 (Python 3.12). The November 2026 "latest" runtime uses
Python 3.13, which does not work with these pinned wheels. The notebook checks
this before installing anything and streams subprocess logs. It keeps data and
runs on mounted Google Drive and calls the same CLI commands listed below. It
also runs in Kaggle with an attached source ZIP and internet access, but save a
notebook version with outputs before exiting because `/kaggle/working` is
ephemeral. This code does not start any paid GPU service.

Until these changes are published, upload the current source archive rather than
an older GitHub clone:

```bash
python NGPS/scripts/export_checkpoint2_source.py --output /tmp/checkpoint2-source.zip
```

The archive leaves out MRI volumes and captures code, tests, configs,
documentation, the notebook, dependency pins and a SHA-256 inventory. Each
trained run also keeps its own source archive. A dirty source snapshot does not
replace the assignment requirement to name a submitted GitHub revision.

Local execution supports CPU, CUDA and Apple Silicon MPS. On a Mac with an
MPS-enabled PyTorch build, pass `--device mps`. The runtime synchronizes MPS
before and after training and inference timing and records current and driver
memory when the backend exposes them. MPS has no portable peak-memory counter,
so the peak field is null. Record the real device and timing hardware in the run
metadata. Do not carry the hosted T4 budget over to a Mac: run a new pilot and
freeze a new workload cap for MPS.

A Mac pilot looks like this:

```bash
python NGPS/scripts/checkpoint2.py pilot \\
  --manifest /path/to/prepared/manifest.json \\
  --config NGPS/configs/checkpoint2.yaml \\
  --output-dir /path/to/mps-pilot --device mps
```

Interactive checkpoint2, training, inference and smoke commands now default to
CUDA when available, then MPS, then CPU. Explicit `--device` arguments are still
better for reproducible experiments. Budget planning can target less than ten
minutes per learned run only after that hardware's pilot. Never report the
target as measured runtime before the pilot and production runs finish.

## Verify or reacquire and prepare

Run from the repository root. Choose new output directories; the commands refuse
to overwrite data or runs. Existing local preparation is in
`outputs/checkpoint2/prepared/manifest.json`, which Git ignores.

To verify the historical sources or reproduce the verified preparation:

```bash
python NGPS/scripts/verify_brainweb.py \
  --manifest NGPS/configs/subjects_v1.json \
  --output /path/to/brainweb_verification.json \
  --cache /path/to/download-cache --accept-fixed-reference
python NGPS/scripts/checkpoint2.py prepare \
  --manifest NGPS/configs/subjects_v1.json \
  --verification /path/to/brainweb_verification.json \
  --output-dir /path/to/prepared
```

On a clean hosted runtime with no original data, reacquire all subjects first:

```bash
python NGPS/scripts/verify_brainweb.py --acquire \
  --manifest /path/to/new-sources/subjects.json \
  --output /path/to/brainweb_verification.json \
  --cache /path/to/download-cache --accept-fixed-reference
python NGPS/scripts/checkpoint2.py prepare \
  --manifest /path/to/new-sources/subjects.json \
  --verification /path/to/brainweb_verification.json \
  --output-dir /path/to/prepared
```

`new-sources` must not exist yet. Failed or partial acquisition is kept, so use
a new source directory for another attempt. The download script records sources,
UTC retrieval time, compressed and decompressed hashes, orientation and
conversion details. Preparation verifies source hashes before normalization. The
output contains ten clean reference arrays, thirty unclipped noisy arrays and a
self-contained manifest. Prepared arrays take about 1.8 GiB, plus source/cache
and model outputs. A reacquired source manifest may have different byte hashes
from another converter; use its matching receipt and prepared manifest throughout
the new experiment.

Fixed split: train `04,06,18,41,42,43`; validation `05,38`; test `20,44`. Each
subject's slices, patches and independent noise draws stay in its split. Sigmas
are `0.03,0.06,0.10`, with shared draws across methods and no clipping of noisy
values or predictions.

## Pilot, freeze and train

```bash
python NGPS/scripts/checkpoint2.py pilot \
  --manifest /path/to/prepared/manifest.json \
  --config NGPS/configs/checkpoint2.yaml --output-dir /path/to/pilot --device cuda
python NGPS/scripts/checkpoint2.py freeze \
  --manifest /path/to/prepared/manifest.json \
  --config NGPS/configs/checkpoint2.yaml --pilot /path/to/pilot/pilot.json \
  --output-dir /path/to/experiment
python NGPS/scripts/checkpoint2.py train \
  --protocol /path/to/experiment/protocol.json --device cuda
```

The pilot uses training subject04 only, sigma0.06, 16 central slices, one epoch
and batch1 for each of the three objectives, and times only one noisy validation
batch. The freeze command projects the full validation cost, applies a 1.5
safety factor and picks one maximum training-batch count per epoch for every
objective. Pass `--max-train-batches N` to freeze a smaller common workload.
Production uses all training subjects and all interior validation slices. Run
budgets measure complete training-command wall time, including setup, guides,
monitoring and saving; they also record GPU training time, validation time,
optimizer steps and peak allocation. The projection is an estimate, and an
observed overrun stops the matrix.

If full validation alone exceeds the cap, reduce the common epoch count in a new
configuration and repeat the pilot, or predeclare a revised cap with
`--budget-seconds SECONDS --budget-reason 'reason'` and a new experiment
directory. Never shorten one model on its own or choose budgets from test
outcomes. The starter config uses width16, four one-block encoder/decoder levels,
128-pixel crops and batch1 as a declared course capacity reduction, shared by
every objective.

All nine runs use seed42, the same initialization, crops, order, optimizer and
workload, separate sigma-specific models, and fixed final-epoch selection.
Noisy-only masked validation is monitoring only. The two trained comparators use
different supervision objectives, and NLM is an additional classical control. NGPS keeps active-pixel-normalized reconstruction and
regional consistency with weight0.5, which differs from the paper's unnormalized
regional sum. Confidence weighting and further ablations are deferred.

A rerun skips runs that match a completed one. Partial runs are preserved and
rejected rather than quietly restarted; use a new experiment to rerun a failed
matrix. Training does not continue after final testing starts. A change to
runtime source, manifests, frozen configuration or checkpoint objective
invalidates preflight, and corrective experiments must stay separate and
reported as such.

## Diagnostics and held-out evaluation

```bash
python NGPS/scripts/checkpoint2.py diagnostics \
  --manifest /path/to/prepared/manifest.json \
  --config /path/to/experiment/training.yaml --subject subject04 --sigma 0.06 \
  --checkpoint /path/to/experiment/runs/ngps_s0.06/model_final.pth \
  --output /path/to/experiment/diagnostics_s0.06.npz
python NGPS/scripts/checkpoint2.py benchmark \
  --protocol /path/to/experiment/protocol.json --device cuda
```

Diagnostics can run before training if you omit `--checkpoint`. The NPZ and PNG
preview contain raw and filtered triplets, static and dynamic masks, patch costs,
selected zero-based neighbor coordinates, retrieved raw values and targets, and
optional predictions. Each run's training history carries the separate NGPS loss
terms and mask statistics. Raw noisy values never get replaced by guides for
targets, and edge coordinates follow replicated boundary handling. Repeat
diagnostics for each sigma.

Benchmark preflight checks all nine final checkpoints, source, config and
protocol fingerprints, completed schedules, budgets and shared test observations
before it opens any held-out reference array. The benchmark saves every
full-volume prediction; image crops are never substituted for volume evaluation.

Expected `experiment/results` outputs:

- `subjects.csv`, `slices.csv`, `aggregates.csv` and `benchmark.json`, with PSNR,
  SSIM (`data_range=1`, uniform 7x7, sample covariance), normalized HFEN (LoG
  sigma1.5, reflect boundaries) and clearly named optional gradient metrics.
- `success.csv`: at each sigma, NGPS mean subject PSNR must meet or exceed the
  stronger blindspot/adjacent comparator with zero tolerance. Unmet targets stay
  failures.
- `worst_slices.csv`: posthoc descriptions of where failures occur, never a
  tuning input.
- Thirty full-volume predictions: five methods or controls × three sigmas × two
  subjects.
- Six matched image grids at frozen central coordinates: full slices, detail
  views and absolute-error maps. Display ranges are common; arrays are not
  clipped.
- `COMPLETE.json` only once the whole benchmark and all figures succeed.

Average slice metrics within each subject, then weight subjects equally. Report
both subjects and the population SD; one training seed and two test subjects
give limited evidence about variability. Pooled slices are not independent
cases. `inference_seconds` covers prediction, transfers and assembly.
`volume_seconds` starts before the noisy file loads and stops after the
prediction is saved. Both exclude model loading, warmup, metrics and hash
verification. Saving measures application I/O, not forced disk durability. NLM
and noisy controls run on CPU, so label their hardware apart from the CUDA
models. Noisy-control time is a copy/load/save control.

## Validation

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m unittest discover -s NGPS/tests -v
python NGPS/scripts/smoke_test.py
python NGPS/scripts/synthetic_pipeline.py --checkpoint2 --work-dir /tmp/new-synthetic-check
```

The test suite includes a tiny synthetic nine-run matrix with 30 full
predictions, all controls, frozen figures, timing, diagnostics and preflight
rejection cases. Synthetic results verify software only. The portable source
archive skips the one historical inventory test when local MRI files are absent.
Notebook output must keep the actual pilot, trained checkpoints, source snapshots
and medical benchmark before anyone claims the Checkpoint 2 experiment is done.
