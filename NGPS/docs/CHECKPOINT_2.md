# Checkpoint 2 implementation and experiment workflow

This checkout integrates runtime code from upstream
`c1ab8bcb2dfa80926c7846048b42356546257f8d` as a source snapshot. The histories
had diverged, so this is not a fast-forward merge. Local README edits and
associated documents are preserved. Historical inventories/splits are retained
in `NGPS/manifests/archive`; use the agreed source manifest for new runs.

## Current evidence

- Ten original compressed MINC files were freshly downloaded from BrainWeb on
  November 11, 2026. Their decompressed contents exactly match the local files.
- The audited MINC real-intensity conversion reproduces all ten historical NumPy
  arrays exactly. Receipt: `NGPS/manifests/brainweb_verification.json`.
- Normalization retains the course fixed scale: MINC real intensities converted
  to float32, clipped to `[0,4095]`, divided by 4095. The MINC stored integer
  range is not its real intensity range. These arrays occupy a subset of `[0,1]`;
  do not describe this as per-subject min/max normalization.
- Source history names `pn3_rf20` coil settings. **These are public fixed reference
  signals, not established noise-free anatomy.** This experiment measures recovery
  after additional Gaussian corruption and cannot establish clinical effectiveness.
- BrainWeb requests attribution; its public pages do not establish an explicit
  redistribution license. Download source volumes directly, exclude them from
  code exports, and do not apply the code MIT license to data.

Sources: [20 normal models](https://brainweb.bic.mni.mcgill.ca/brainweb/anatomic_normal_20.html),
[subject 04 download information](https://brainweb.bic.mni.mcgill.ca/cgi/brainweb1?alias=subject04_t1w&download=1),
[MINC/raw format explanation](https://brainweb.bic.mni.mcgill.ca/brainweb/about_data_formats.html),
[requested references](https://brainweb.bic.mni.mcgill.ca/brainweb/#references).

## Environment and hosted execution

Use Python 3.11 or 3.12 in an isolated environment. Install the pinned hosted
runtime with `python -m pip install -r NGPS/requirements-colab.txt` before
importing PyTorch. The existing general requirements remain available for CPU
checks. Record actual versions/hardware from generated run metadata.

Open `NGPS/notebooks/checkpoint2_colab.ipynb` in Colab. In Runtime > Change runtime
type, select T4 and Runtime version **2026.07 (Python 3.12)**. The November 2026
latest runtime uses Python 3.13, which is incompatible with these pinned wheels.
The notebook checks this before installation and streams subprocess logs.
The notebook uses mounted Google Drive for persistent data/runs, and calls the
same CLI commands listed below. It also works in Kaggle using an attached source
ZIP and internet access; **save a notebook version with outputs before exiting**
because `/kaggle/working` is ephemeral. No paid GPU service is started by this code.

Until these changes are published, upload the current source archive, not an
older GitHub clone:

```bash
python NGPS/scripts/export_checkpoint2_source.py --output /tmp/checkpoint2-source.zip
```

The archive excludes MRI volumes and captures code, tests, configs, documentation,
notebook, dependency pins, and an accompanying SHA-256 inventory. Each trained run
also retains its own source archive. A dirty source snapshot does not replace the
assignment's eventual requirement to name a submitted GitHub revision.

The local implementation environment has CPU-only PyTorch; it cannot establish
GPU performance. Medical training runtime is determined by the hosted pilot.
Budget planning targets less than ten minutes per learned run, initially nine
runs total, excluding NLM/evaluation/data acquisition. Never report this target as
measured runtime before the pilot and production runs complete.

## Verify or reacquire and prepare

Run from the repository root. Choose new output directories; commands refuse to
overwrite data/runs. The existing local preparation is in
`outputs/checkpoint2/prepared/manifest.json` (ignored by Git).

To verify existing historical sources, or reproduce the verified preparation:

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

`new-sources` must not exist. Failed/partial acquisition is retained; use a new
source directory for a new attempt. The download script records sources, UTC
retrieval, compressed/decompressed hashes, orientation, and conversion details.
Preparation verifies source hashes before normalization. Output contains ten clean
reference arrays, thirty unclipped noisy arrays, and a self-contained manifest.
Disk usage is about 1.8 GiB for prepared arrays plus source/cache and model outputs.
A reacquired source manifest may have different byte hashes from another converter;
use its matching receipt and prepared manifest throughout a new experiment.

Fixed split: train `04,06,18,41,42,43`; validation `05,38`; test `20,44`.
Each subject's slices, patches, and independent noise draws stay in its split.
Sigmas: `0.03,0.06,0.10`; shared draws across methods; no noisy or prediction clipping.

## Pilot, freeze, and train

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

The pilot uses only training subject04, sigma0.06, 16 central slices, one epoch,
batch1, for each of the three objectives; only one noisy validation batch is timed.
The freeze command projects the full validation cost, uses a 1.5 safety factor,
and chooses an identical maximum training-batch count per epoch. Supply
`--max-train-batches N` to freeze a smaller common workload. Production uses all
training subjects and all interior validation slices. Per-run budgets measure
complete training-command wall time, including setup, guides, monitoring and saving.
GPU training time, validation time, optimizer steps, and peak allocation are also
recorded. The projection is an estimate; an observed budget overrun stops the matrix.

If full validation alone exceeds the cap, reduce the common epoch count in a new
configuration and repeat the pilot, or predeclare a revised cap with both
`--budget-seconds SECONDS --budget-reason 'reason'`. Use a new experiment directory.
Never silently shorten individual models or use test outcomes to choose budgets.
The starter config uses width16, four one-block encoder/decoder levels, 128-pixel
crops and batch1 as a declared course capacity reduction; every objective shares it.

All nine runs use seed42, equal initialization/crops/order/optimizer/workload,
separate sigma-specific models, and fixed final-epoch selection. Noisy-only
masked validation is monitoring only. The two trained comparators are genuinely
distinct supervision objectives. NLM remains an additional classical control.
NGPS retains active-pixel-normalized reconstruction and regional consistency with
weight0.5, explicitly different from the paper's unnormalized regional sum.
Confidence weighting and additional ablations are deferred.

Completed matching runs are skipped on rerun. Partial runs are preserved and
rejected, rather than silently restarted. Use a new experiment to rerun a failed
matrix. Training does not continue after final testing starts. Changing runtime
source, manifests, frozen configuration, or checkpoint objective invalidates
preflight; corrective experiments must remain separate and be reported honestly.

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

Diagnostics can run before training by omitting `--checkpoint`. The NPZ and a PNG preview contain
raw/filtered triplets, static/dynamic masks, patch costs, selected zero-based
neighbor coordinates, retrieved raw values/targets, and optional predictions.
Separate NGPS loss terms and mask statistics appear in each run's training history.
Raw noisy values are never replaced by guides for targets. Edge coordinates follow
replicated boundary handling. Repeat diagnostics for each sigma.

Benchmark preflight checks all nine final checkpoints, source/config/protocol
fingerprints, completed schedules, budgets and shared test observations before any
held-out reference arrays are opened. The benchmark saves every full-volume
prediction; no image crops are substituted for volume evaluation.

Expected `experiment/results` outputs:

- `subjects.csv`, `slices.csv`, `aggregates.csv`, and `benchmark.json` with PSNR,
  SSIM (`data_range=1`, uniform7x7/sample covariance), normalized HFEN (LoG sigma1.5,
  reflect boundaries), and explicitly named optional gradient metrics.
- `success.csv`: at each sigma, NGPS mean subject PSNR must meet/exceed the stronger
  blindspot/adjacent comparator with **zero tolerance**. Unmet targets remain failures.
- `worst_slices.csv`: explicitly posthoc descriptive failure locations, never tuning.
- Thirty full-volume predictions: five methods/controls × three sigmas × two subjects.
- Six matched image grids at frozen central coordinates: full slices, detail views,
  and absolute-error maps. Display ranges are common; arrays are not clipped.
- `COMPLETE.json` only after the complete benchmark and all figures succeed.

Average slice metrics within each subject, then weight subjects equally. Report
both subjects and population SD; one training seed and two test subjects provide
limited variability evidence. Do not interpret pooled slices as independent cases.
`inference_seconds` includes prediction/transfers/assembly; `volume_seconds` starts
before noisy file loading and stops after prediction saving. Both exclude model
loading, warmup, metrics and hash verification. File saving measures application
I/O, not forced disk durability. NLM and noisy controls run on CPU; label their
hardware separately from CUDA models. Noisy-control time is a copy/load/save control.

## Validation

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m unittest discover -s NGPS/tests -v
python NGPS/scripts/smoke_test.py
python NGPS/scripts/synthetic_pipeline.py --checkpoint2 --work-dir /tmp/new-synthetic-check
```

The test suite includes a tiny synthetic nine-run matrix with 30 full predictions,
all controls, frozen figures, timing, diagnostics and preflight rejection cases.
Synthetic results verify software only. The portable source archive skips the one
historical inventory test when local MRI files are absent. Notebook output must
retain the actual pilot, trained checkpoints, source snapshots and medical benchmark
before claiming Checkpoint 2 experiment completion.
