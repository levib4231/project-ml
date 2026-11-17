# Final verification

Verified locally on Apple Silicon, Python3.13.13, PyTorch2.13.0 with MPS available.
This is the actual local environment, not a claim that the pinned historical
Python3.12 Colab wheels work on Python3.13. Historical environments remain in
the corresponding run metadata/source snapshots.

Completed:

- `python -m compileall -q NGPS`
- `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m unittest discover -s NGPS/tests -v`
  —44 tests passed.
- CPU and MPS `NGPS/scripts/smoke_test.py` passed.
- Standard synthetic train/infer/evaluate pipeline passed.
- Synthetic nine-run Checkpoint2 benchmark passed.
- `git diff --check` passed.
- `NGPS/scripts/package_review.py --verify` passed:231 files,5226016 bytes.
- `NGPS/scripts/verify_review.py --local` passed:three complete benchmarks,
  90 subject rows,27 checkpoint hashes,90 prediction hashes,100 local data-file
  hash checks (including repeated shared-source checks),three historical source
  trees, and31 noisy-only/ablation production summaries checked against metadata.

`numerical_audit.json` records the final runtime source-tree fingerprint. It does
not replace historical hashes. The current runtime cannot silently resume a
historical protocol after source changes; use that run's archived source.

The source fingerprint covers runtime, configs, tests and NGPS documentation,
not this repository-root evidence directory. Final commit identity is obtained
with `git rev-parse HEAD`; embedding a commit's own hash in its contents would
be self-referential. Dirty historical runs are recoverable from source snapshots,
not from commit IDs alone.

The reporting-only failure/recovery in noise isolation and the rejected MPS
budget freeze are disclosed in the evidence guide and packaged study records.
No historical model was retrained just to make documentation/source hashes match.

Logs are copied verbatim. Temporary synthetic directories identify local test
artifacts, not required downloadable datasets. No public action was performed.
