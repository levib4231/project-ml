# NGPS: Neighbor-Guided Patch Sampling

Local correctness and reproducibility work for the CPSC T3 assignment in
[levib4231/project-ml](https://github.com/levib4231/project-ml). We train a 2D
NAFNet on noisy adjacent slices and add guide-based matching in dynamic regions.
This is not a verified reproduction of published results and not a clinical
tool.

## Submission status

The original frozen BrainWeb/Gaussian T4 benchmark remains primary: NGPS missed
the predeclared PSNR criterion at all three sigmas. Later MPS/Rician/ablation
findings are exploratory and do not replace that result. Read
[Submission evidence](docs/SUBMISSION_EVIDENCE.md) for the evidence hierarchy,
small artifact package, historical source/access instructions and final checks.

## Install

```bash
git clone https://github.com/levib4231/project-ml.git
cd project-ml/NGPS
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For CPU-only PyTorch, install `torch` from
`https://download.pytorch.org/whl/cpu` first. For CUDA, follow the official
PyTorch instructions. On Apple Silicon, use a PyTorch build with MPS support and
pass `--device mps`; the project synchronizes MPS for timing and records
available MPS memory telemetry. CPU CI uses Python 3.12. Conversion dependencies
and normalization caveats are in [DATA](docs/DATA.md).

## Self-contained synthetic check

From `NGPS/`, with a fresh output directory outside the repository:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python scripts/synthetic_pipeline.py \
  --work-dir /tmp/project-ml-synthetic-run
```

The script creates ten distinct toy subjects, a 6/2/2 subject split and all three
noise levels. It trains a tiny NGPS at the default sigma 0.06 for one fixed
epoch, infers a complete volume and writes held-out model and NLM metrics in
JSON/CSV. Treat it as a pipeline check, not a benchmark. The older
`examples/train_synthetic_demo.*` folder-based commands are still in the
repository for reference but no longer match the manifest-based training path.

## Prepare assignment data

`configs/subjects_v1.json` fixes the ten BrainWeb subject identities:

- train: 04, 06, 18, 41, 42, 43
- validation: 05, 38
- test: 20, 44

The historical conversion was verified against downloaded MINC files; its
course /4095 scale is not per-volume min/max normalization or a guarantee of
noise-free anatomy. Use the verified Checkpoint 2 preparation workflow for the
medical benchmark. The generic preparation command below still requires
explicitly justified normalization and never infers min/max. For inputs
independently confirmed to be in `[0,1]`:

```bash
python scripts/prepare_data.py --manifest configs/subjects_v1.json \
  --output-root /absolute/path/prepared-v1 --normalization already-unit
```

Otherwise use `--normalization fixed-range --lower L --upper U`, replacing `L`
and `U` with justified bounds declared before the run, not bounds derived from
test data. Preparation rejects out-of-range sources. [DATA](docs/DATA.md) has the
full contract.

## Train: same backbone, different objectives

```bash
python scripts/train_ngps.py --manifest /absolute/path/prepared-v1/manifest.json \
  --sigma 0.06 --method ngps --config configs/ngps_nafnet_default.yaml \
  --output-dir /absolute/path/runs/ngps-s006 --device cpu
```

Repeat separately for sigma 0.03, 0.06 and 0.10. Replace `--method` with:

| Method | Training objective |
|---|---|
| `ngps` | static adjacent targets, guide-matched Top-K dynamic targets, regional consistency |
| `adjacent` | unmasked same-coordinate previous/next noisy targets; no guide, no consistency term |
| `blindspot` | randomly replace 5% of noisy pixels; MSE only at replaced coordinates; no adjacent targets |
| `nsn2n` | static-region adjacent reconstruction plus regional and interpolation consistency |

`train_nsn2n.py` is also an NS-N2N entry point. NLM needs no training. All neural
methods use a predeclared final-epoch checkpoint and never select on clean
references. Fixed masked noisy validation is monitoring only. Run metadata
records the config, seeds, full manifest and hash, commit and dirty flag,
environment, epoch, mask and matching diagnostics, and validation history. See
[REPRODUCIBILITY](docs/REPRODUCIBILITY.md).

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

Evaluation opens test subjects only, checks data hashes and checkpoint split
provenance, and processes every slice, including boundary slices. NLM runs all
three sigmas unless you pass `--sigma`. Reports contain per-subject metrics,
equal-subject aggregate means and population SD per sigma, and complete-volume
wall time. Outputs are `.json`, `.csv` and `_aggregate.csv`.

Metrics: PSNR (`data_range=1`), SSIM, normalized LoG HFEN, GMSD and
`SobelGradientMagnitudeSimilarity`. The gradient metric is not FSIM. Neither
observations nor predictions are clipped. A standalone
`evaluate_volume.py --pred ... --clean ... --output-csv ...` is still available,
but it does not enforce the held-out protocol.

## Tests

From the repository root:

```bash
python -m compileall -q NGPS
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python NGPS/scripts/smoke_test.py
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m unittest discover -s NGPS/tests -v
git diff --check
```

[Method mapping and NAFNet comparison](docs/METHOD_MAPPING.md) documents changed
architecture semantics. Old checkpoints must be retrained; inference rejects
them rather than guessing at the new global residual.
