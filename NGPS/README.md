# NGPS: Neighbor-Guided Patch Sampling

Local correctness/reproducibility implementation for the CPSC T3 assignment in
[levib4231/project-ml](https://github.com/levib4231/project-ml). A 2D NAFNet is
trained on noisy adjacent slices, with guide-based matching for dynamic regions.
This is **not** a verified reproduction of published results or a clinical tool.

## Install

```bash
git clone https://github.com/levib4231/project-ml.git
cd project-ml/NGPS
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For CPU-only PyTorch, install `torch` from
`https://download.pytorch.org/whl/cpu` first. For CUDA use the official PyTorch
installation instructions. Python 3.12 is used in CPU CI. Optional MINC/NIfTI
conversion dependencies and normalization caveats are in [DATA](docs/DATA.md).

## Self-contained synthetic check

From `NGPS/`, with a fresh output directory **outside** the repository:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python scripts/synthetic_pipeline.py \
  --work-dir /tmp/project-ml-synthetic-run
```

This creates ten distinct toy subjects, a 6/2/2 subject split, and all three
noise levels; it trains a tiny NGPS at the default sigma 0.06 for one fixed
epoch, infers a complete volume, and writes held-out model and NLM metrics in
JSON/CSV. It is a pipeline check, not a
benchmark. The older `examples/train_synthetic_demo.*` folder-based commands are
legacy and are **superseded by this example**; training now requires a manifest.
Those files are outside this pass's permitted edit scope.

## Prepare assignment data

`configs/subjects_v1.json` fixes the ten existing BrainWeb subject identities:

- train: 04, 06, 18, 41, 42, 43
- validation: 05, 38
- test: 20, 44

**The existing BrainWeb intensity scale has not been verified.** Inspect source
metadata/conversion provenance before choosing a normalization. Preparation never
infers min/max or silently clips. If inputs have independently been confirmed to
be in `[0,1]`:

```bash
python scripts/prepare_data.py --manifest configs/subjects_v1.json \
  --output-root /absolute/path/prepared-v1 --normalization already-unit
```

Otherwise use `--normalization fixed-range --lower L --upper U`, replacing `L`
and `U` with justified, predeclared bounds (not test-derived bounds). Preparation
rejects out-of-range sources. See [DATA](docs/DATA.md) for the full contract.

## Train: same backbone, distinct objectives

```bash
python scripts/train_ngps.py --manifest /absolute/path/prepared-v1/manifest.json \
  --sigma 0.06 --method ngps --config configs/ngps_nafnet_default.yaml \
  --output-dir /absolute/path/runs/ngps-s006 --device cpu
```

Repeat separately for sigma `0.03`, `0.06`, and `0.10`. Replace `--method` with:

| Method | Actual training objective |
|---|---|
| `ngps` | static adjacent targets + guide-matched Top-K dynamic targets + regional consistency |
| `adjacent` | unmasked same-coordinate previous/next noisy targets; no guide, no consistency term |
| `blindspot` | randomly replace 5% of noisy pixels; MSE only at replaced coordinates; no adjacent targets |
| `nsn2n` | static-region adjacent reconstruction + regional and interpolation consistency |

`train_nsn2n.py` is also an NS-N2N entry point. NLM requires no training. All
neural methods use a **predeclared final-epoch checkpoint**, never clean-reference
selection. Fixed masked noisy validation is monitoring only. Run metadata records
config, seeds, full manifest/hash, commit/dirty flag, environment, epoch, mask and
matching diagnostics, and validation history. See [REPRODUCIBILITY](docs/REPRODUCIBILITY.md).

## Inference and held-out evaluation

```bash
python scripts/infer_volume.py --checkpoint /absolute/path/runs/ngps-s006/model_final.pth \
  --input /absolute/path/noisy.npy --output /absolute/path/prediction.npy --device cpu
python scripts/evaluate_manifest.py --manifest /absolute/path/prepared-v1/manifest.json \
  --method model --checkpoint /absolute/path/runs/ngps-s006/model_final.pth \
  --sigma 0.06 --output-prefix /absolute/path/results/ngps-s006 --device cpu
python scripts/evaluate_manifest.py --manifest /absolute/path/prepared-v1/manifest.json \
  --method nlm --output-prefix /absolute/path/results/nlm
```

Evaluation opens **test subjects only**, checks data hashes and checkpoint split
provenance, and processes **every slice**, including boundary slices. NLM runs all
three sigmas unless `--sigma` is supplied. Reports contain per-subject metrics,
equal-subject aggregate means/population SD per sigma, and complete-volume wall
time. Outputs: `.json`, `.csv`, and `_aggregate.csv`.

Metrics: PSNR (`data_range=1`), SSIM, normalized LoG HFEN, explicitly named
`SobelGradientMagnitudeSimilarity`, and GMSD. The gradient metric is **not FSIM**.
No observation or prediction clipping is performed. A standalone
`evaluate_volume.py --pred ... --clean ... --output-csv ...` remains available,
but does not enforce the held-out protocol.

## Tests

From the repository root:

```bash
python -m compileall -q NGPS
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python NGPS/scripts/smoke_test.py
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m unittest discover -s NGPS/tests -v
git diff --check
```

[Method mapping and NAFNet comparison](docs/METHOD_MAPPING.md) documents changed
architecture semantics. Old checkpoints must be retrained; they are rejected
rather than silently interpreted with the new global residual.
