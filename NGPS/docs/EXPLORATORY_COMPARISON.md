# Noisy-only comparison pilot

This follow-up leaves the completed T4 and MPS medical benchmarks alone. Neither
showed the predeclared NGPS advantage. The existing test subjects are already
exposed, so any later scoring on them is exploratory.

## Implemented first stage

`ngps/comparison.py` and `scripts/comparison_pilot.py` run a bounded feasibility
study. They do not claim better denoising quality. The setup:

- training observations from subject04 only, sigma 0.06;
- 32 central training slices, batch1, width16, original optimizer, seed42;
- independently initialized matched 32- and 64-update schedules (1 and 2 epochs);
- full NGPS, blindspot, adjacent, NGPS without RC and NGPS without the dynamic
  loss;
- the first 16 noisy validation batches with fixed masks, for monitoring only;
- a fixed final checkpoint, with no ranking, tuning or winner selection from
  scores;
- an independently declared MPS pilot safety cap of 600 seconds per run, not a
  reused T4 performance budget. Record measured times and stop on overrun.

Every variant keeps the same initialization, shuffled samples, deterministic
crops and optimizer schedule. The two ablations keep the original active-pixel
normalization, omit only the designated term, and do not rescale the rest. They
still compute matching and neighbor forwards, so their timings do not estimate
what an optimized implementation would save. Training losses differ by objective
and must not be ranked as denoising quality.

The runner freezes a hashed protocol before training, refuses existing output
directories, records source snapshots per run and keeps partial evidence when a
run fails. New ablations cannot pass the frozen Checkpoint 2 validator as full
NGPS. Default training behavior is unchanged.

Opt-in gradient diagnostics observe the L2 norm without clipping, fail on
nonfinite norms and have a regression test for identical CPU model updates. One
fixed training crop supplies all-region and dynamic-region cost quantiles,
clamped retrieval-offset distances, Top-K spatial spread and mask fractions.
That crop is a local diagnostic, not population evidence.

```bash
python NGPS/scripts/comparison_pilot.py \
  --manifest outputs/checkpoint2/medical/mps_full_prepared/manifest.json \
  --config outputs/checkpoint2/medical/mps_full_run/training.yaml \
  --output-dir outputs/checkpoint2/exploratory/comparison_pilot_v1 \
  --device mps
```

## Next stages

None of these are implemented or authorized by the pilot outcomes:

1. Predeclare full-training matched 531/1062/2148-update convergence
   experiments, pilot their MPS cost and freeze realistic workload caps before
   running them.
2. Run full-workload loss ablations and keep each variant separately named.
3. Compare one larger capacity across all required trained objectives.
4. Consider confidence weighting only as a separately tested method change.
5. Freeze all final candidates before scoring, and get untouched subjects for
   independent confirmation. Never use clean arrays for tuning or selection.

The experiment index lives under `outputs/checkpoint2/exploratory/`. Frozen
historical outputs must not be rewritten to accommodate source changes.
