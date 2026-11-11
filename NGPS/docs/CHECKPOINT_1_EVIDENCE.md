# Checkpoint 1 – Reproducibility and Initial Evidence

This document records the first verified implementation checkpoint for the NGPS project. It is evidence of a working, reproducible starting point, not a scientific performance claim.

## Repository and scope

- Repository: `https://github.com/levib4231/project-ml`
- Base commit inspected: `1113ee1` – initial commit
- Subject manifest: `NGPS/configs/subjects_v1.json`
- Manifest contents: 10 subjects, deterministic 6 train / 2 validation / 2 held-out test split
- Forbidden source/output scope checked: no `Data/Clean/`, PDF, or `Figure_1.png` changes
- Clean-reference training/selection: documented as prohibited by the implementation protocol

## Reproduced verification

The commands below were run from the repository root. Runtime tests used the existing project environment at `/tmp/project-ml-venv/bin/python`, which contains the project’s Torch, OpenCV, pytest, scikit-image, and NumPy dependencies.

### Static checks

```text
python -m compileall -q NGPS
PASS

git diff --check
PASS
```

### Test suite

```text
/tmp/project-ml-venv/bin/python -m pytest NGPS/tests -q
27 passed, 170 subtests passed in 3.54s
```

The tests cover deterministic subject splitting and seeds, manifest leakage, data validation and normalization, unclipped noise generation, guide and mask behavior, sampler bounds and diagnostics, NAFNet residual behavior, loss arithmetic, baselines, subject-level metrics, configuration validation, and a synthetic end-to-end path.

### Existing model smoke test

```text
/tmp/project-ml-venv/bin/python NGPS/scripts/smoke_test.py
{'loss': 0.4204060733318329, 'recon': 0.4144190549850464, 'n2n': 0.1892201453447342, 'dynamic': 0.2251989245414734, 'rc': 0.011974053457379341, 'ic': 0.0}
```

The smoke test completed with exit code 0.

### Synthetic train → inference → held-out evaluation

```text
/tmp/project-ml-venv/bin/python NGPS/scripts/synthetic_pipeline.py \
  --work-dir <fresh scratch directory>
```

Observed output:

```text
{'epoch': 1,
 'loss': 0.051465207089980446,
 'dynamic_active_fraction': 0.6021412114302317,
 'dynamic_empty_fraction': 0.0,
 'static_active_fraction': 0.39785878856976825,
 'static_empty_fraction': 0.0,
 'match_cost_mean': 0.007230296575774749,
 'match_cost_max': 0.1115669384598732,
 'noisy_validation_mse': 0.031628023250959814}
Synthetic train/infer/evaluate PASS: 2 held-out subjects, standalone volume inference 0.002382s
```

This confirms the code can construct synthetic data, train without clean-reference checkpoint selection, run standalone volume inference, and produce held-out evaluation artifacts for two subjects.

## What this does and does not establish

Established by the checks above:

- the repository compiles;
- the new tests pass in the project runtime environment;
- deterministic manifest/noise/data-validation behavior is exercised;
- the training, inference, and evaluation path runs end to end on synthetic data;
- the held-out split is present and separated by subject;
- sampler diagnostics and active/empty mask fractions are observable.

Not established yet:

- BrainWeb intensity scale, orientation, and conversion provenance;
- medical-data training results;
- superiority over published baselines;
- clinical usefulness or downstream-task benefit;
- exact reproduction of the NGPS paper’s CT or synthetic-Rician MRI numbers.

The next evidence step is to verify the real BrainWeb conversion and run the same manifest-based protocol on the actual held-out subjects before making any scientific claim.
