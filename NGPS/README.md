# NGPS in Project ML

This package contains a course-scale Neighbor-Guided Patch Sampling (NGPS)
implementation plus reproducibility tooling. It is not an unqualified
reproduction of the paper: the medical dataset identity, final corruption
protocol, required baselines, and compute target remain unresolved.

## Status

- The NGPS/NAFNet synthetic training, inference, and evaluation path is
  implemented.
- Training and checkpoint selection use noisy train/validation volumes only.
- Dataset inventory, hashing, deterministic subject splits, configurable
  Rician corruption, run metadata, and subject-first evaluation are implemented.
- The existing comparison code is labeled **NS-N2N-style**; equivalence to a
  selected external implementation has not been validated.
- NLM, a masked-pixel blind-spot baseline, a naïve neighboring-slice baseline,
  and confidence gating are not implemented.
- `fsim_proxy` is a lightweight gradient-similarity proxy, not reference FSIM.
- No medical benchmark results are claimed.

See [PROJECT_SCOPE.md](docs/PROJECT_SCOPE.md),
[DATA_PROVENANCE.md](docs/DATA_PROVENANCE.md), and
[NOISE_PROTOCOL.md](docs/NOISE_PROTOCOL.md).

## Installation

```bash
git clone https://github.com/levib4231/project-ml.git
cd project-ml
python -m venv .venv
source .venv/bin/activate
python -m pip install -r NGPS/requirements.txt
python -m pip install -r NGPS/requirements-dev.txt
```

Install a CUDA-specific PyTorch build from the official PyTorch instructions
when GPU training is required.

## Synthetic pipeline

The synthetic dataset contains simple moving shapes with additive Gaussian
noise. It is software verification, not a medical benchmark.

```bash
python NGPS/scripts/make_synthetic_demo_data.py --output-root /tmp/project-ml-synthetic
python NGPS/scripts/train_ngps.py \
  --data-root /tmp/project-ml-synthetic \
  --config NGPS/configs/synthetic_demo.yaml \
  --output-dir /tmp/project-ml-ngps \
  --device cpu
python NGPS/scripts/infer_volume.py \
  --checkpoint /tmp/project-ml-ngps/model_best.pth \
  --input /tmp/project-ml-synthetic/val/noisy/val_case_000.npy \
  --output /tmp/project-ml-ngps/val_case_000.npy \
  --device cpu \
  --clip-output
python NGPS/scripts/evaluate_volume.py \
  --pred /tmp/project-ml-ngps/val_case_000.npy \
  --clean /tmp/project-ml-synthetic/val/clean/val_case_000.npy \
  --output-csv /tmp/project-ml-ngps/val_case_000_slices.csv \
  --output-json /tmp/project-ml-ngps/val_case_000_metrics.json
```

Each training run writes `run_manifest.json`, `validation.jsonl`,
`model_last.pth`, and the noisy-validation-selected `model_best.pth`.

Use `--dry-run` to validate the config, normalized noisy volumes, triplet
construction, and model construction without optimizing:

```bash
python NGPS/scripts/train_ngps.py \
  --data-root /path/to/DATA_ROOT \
  --config NGPS/configs/ngps_nafnet_default.yaml \
  --output-dir /tmp/ngps-dry-run \
  --device cpu \
  --dry-run
```

## Data contract and isolation

Training expects normalized float volumes with shape `(D,H,W)`:

```text
DATA_ROOT/
  train/noisy/*.npy
  val/noisy/*.npy
```

Values outside `[0,1]`, non-finite values, non-floating dtypes, and unexpected
dimensions are rejected instead of silently clipped. Training inventory and
`VolumeTripletDataset` objects contain no clean path. Clean references are
opened only by explicit evaluation tools.

The committed medical files are not silently wired into training. First create
and review an inventory and split manifest; their identity and license are still
unverified:

```bash
python NGPS/scripts/make_manifest.py \
  --data-dir Data/Clean \
  --dataset-name committed-subject-volumes-identity-unverified \
  --preprocessing-version convert_brain.py-v1 \
  --provenance-note "identity and license unresolved" \
  --output /tmp/inventory.json
python NGPS/scripts/make_split_manifest.py \
  --manifest /tmp/inventory.json --counts 6 2 2 --seed 42 \
  --output /tmp/split.json
```

## Configurable Rician simulation

The simulator uses
`sqrt((x+n_real)^2+n_imag^2)` with independent Gaussian components. The sigma
reference must be specified; output clipping is opt-in:

```bash
python NGPS/scripts/generate_rician_noise.py \
  --manifest /tmp/split.json \
  --output-root /tmp/rician \
  --sigma 0.05 \
  --sigma-reference data_range_fraction \
  --clip-output
```

That command defines sigma as 5% of the configured data range. It does **not**
claim to reproduce the paper’s “5%” convention. The corruption manifest stores
the equation, effective sigma, clipping choice, and per-subject seed.

## Evaluation

`evaluate_volume.py` writes slice-level intermediate metrics and one subject
summary. `evaluate_dataset.py` averages slices within each subject and then
reports mean, sample standard deviation, and a 95% Student-t interval across
subjects. Its default evaluation region is the full image, and metrics are
computed on 2D slices.

PSNR and SSIM are project implementations with `data_range=1`; HFEN and GMSD
have not yet been cross-validated against an approved course reference. The
feature-similarity output is deliberately named `fsim_proxy`.

## Method configuration

The default project configuration uses patch size 7, search window 15, Top-K
4, mask threshold 0.05, regional-consistency weight 0.5, and AdamW with learning
rate 2e-4. These match values stated in the local paper, but the surrounding
dataset/preprocessing pipeline is not yet a paper reproduction.

## Citation

The local paper supplies the title and authors below. Verify final venue details
before using this in a submitted report:

```bibtex
@article{cho2026ngps,
  title  = {NGPS: Structure-Preserving Self-Supervised Denoising via Neighbor-Guided Patch Sampling},
  author = {Cho, Jaehyun and Yoo, YoungJoon},
  year   = {2026},
  note   = {Local preprint; final publication metadata must be verified}
}
```
