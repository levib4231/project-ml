# Submission evidence and limits

## Primary result: original frozen Gaussian benchmark

The primary Checkpoint2 experiment is the completed BrainWeb/Gaussian T4 run:
seed42, width16, one fixed final epoch,555 updates per learned run, nine learned
runs, agreed6/2/2 split, sigmas0.03/0.06/0.10. Test subjects20/44 have equal weight.
NGPS had to match or exceed the stronger trained comparator at **every sigma**,
with zero PSNR tolerance. It did not meet that criterion.

| Sigma | NGPS mean subject PSNR | Adjacent PSNR | Gap(dB) |
|---|---:|---:|---:|
| 0.03 | 37.5392 | 38.3301 | -0.7909 |
| 0.06 | 35.3161 | 36.0157 | -0.6996 |
| 0.10 | 34.0128 | 34.7421 | -0.7293 |

These results remain primary. No later experiment overwrites, repairs, relabels,
or supersedes the unmet criterion. Recovery of fixed reference signals after
added noise is not evidence of noise-free anatomy or clinical effectiveness.

## Separately labeled follow-ups

- **MPS/Gaussian benchmark:** independent hardware pilot, one epoch,531 updates;
  same methods/sigmas/split. NGPS missed the criterion at all three sigmas.
- **Noisy-only convergence:** sigma0.06,531/1062/2148 updates, NGPS/blindspot/
  adjacent. NGPS monitoring improved modestly; no clean/test scoring occurred.
- **Loss ablations:** separately named noRC/noDynamic variants. Original default
  NGPS behavior was not changed. Noisy-only results are not clean-risk rankings.
- **BrainWeb/Rician:** exact synthetic magnitude corruption, nine fresh531-update
  models. NGPS beat required trained comparators only at sigma0.06 (+0.4463dB);
  overall criterion still unmet. HFEN did not favor NGPS over adjacent there.
- **Paired-noise isolation:** sigma0.06,531 updates. Dynamic-loss contribution
  (full minus noDynamic PSNR) was -0.9990dB under Gaussian and+0.3866dB under
  Rician. About87% of Rician volume-MSE gain vs adjacent occurred at reference
  intensity<0.02. This is consistent with reduced low-signal positive error, not
  proof of better anatomical correspondence or stochastic bias correction.

All subsequently scored subjects were already exposed. These are exploratory
findings, not independent confirmation. They are not paper reproduction: the
paper used IXI/Rician, different training/capacity and a different RC normalization.
No clean reference was used to train, validate, tune, early-stop or select a
checkpoint. Choices influenced by earlier test findings remain exploratory even
when later protocols were frozen before scoring.

## Read the evidence

The repository-root `evidence/checkpoint2/` package contains copied small records
under `records/outputs/checkpoint2/`, an SHA-256 inventory, and deduplicated
historical code snapshots. It excludes medical arrays, figures, predictions and
weights. Full per-study reports are included alongside original protocols,
configs, summaries, access audits and run metadata. Original outputs are not edited.

Historical source fingerprints intentionally differ from the final runtime.
Use each run's archived source to reproduce it. Do not change historical hashes
or bypass current-source validation to make an old protocol appear current.
Paths inside copied manifests/metadata retain their original local/Colab meanings;
the review copy is not a relocated executable dataset. Historical scripts must
be copied back to their documented `outputs/...` locations and use compatible
archived runtime/data layout. Do not execute them inside the evidence package.

### Data and checkpoint access

BrainWeb volumes are not licensed by the code's MIT license. Obtain them directly
using `NGPS/scripts/verify_brainweb.py` and the documented verification workflow.
No source volume is bundled. Historical preparation references the explicit
course /4095 scale, not per-subject min/max normalization. Downloaded/generated
byte hashes can differ if converter/NumPy versions differ; keep the matching
receipt, manifest and runtime for a new experiment.

Large local artifacts remain in `outputs/checkpoint2/medical/` and
`outputs/checkpoint2/exploratory/`. The original T4 archive's existing team Drive
link is in the root README. MPS/exploratory weights and predictions are local
only; request them from the project owner for private transfer. No new public
hosting or upload has been performed, and their remote availability is not claimed.

## Failures and recovery

- The MPS ten-epoch/600-second freeze was rejected before production because
  projected validation/setup exceeded the cap; a separate one-epoch schedule was
  subsequently frozen. The original T4 workload decision is also retained.
- Paired-noise isolation finished all training/scoring but failed summary
  aggregation on `subject` versus `subject_id`. A separate finalizer repaired
  reporting only, with original script/scores/log preserved locally. No models
  were retrained or rescored. `reporting_recovery.json` records the hashes.
- Partial attempts remain local; this small review package includes the failure
  narratives and recovery records, not every partial-log duplicate.

## Final checks and local packaging

From the repository root:

```bash
python -m compileall -q NGPS
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m unittest discover -s NGPS/tests -v
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python NGPS/scripts/smoke_test.py --device cpu
python NGPS/scripts/synthetic_pipeline.py --work-dir /tmp/ngps-final-standard
python NGPS/scripts/synthetic_pipeline.py --checkpoint2 --work-dir /tmp/ngps-final-matrix
python NGPS/scripts/package_review.py --verify --output evidence/checkpoint2
python NGPS/scripts/verify_review.py --local --output /tmp/ngps-review-audit.json
git diff --check
```

Synthetic directories and audit paths must be new. `--local` additionally verifies
large artifacts available on the owner's machine; a reviewer without volumes can
omit it. Numerical audit recomputes subject aggregates and checks CSV/JSON tables,
zero-tolerance criterion, protocol/config hashes, historical source snapshots and
completed workload/budget evidence. Current code is checked separately; medical
models are not silently rerun for documentation/refactoring changes.

The small evidence package can be regenerated to a new directory with
`python NGPS/scripts/package_review.py --output /path/to/new-review`. Verify it
before replacing a review copy; never use regeneration to rewrite original runs.
A separate local source ZIP can be made with `export_checkpoint2_source.py`.
It excludes the repository-root evidence, which must be included separately.
No packaging command uploads anything.

Report/slides/video/Canvas ZIP remain separate deliverables. Public upload or
push requires the owner's explicit confirmation immediately before that action.
