# Reproducibility Contract

Each training run writes a `run_manifest.json` containing the command, complete
configuration, random seed, Git commit and dirty-worktree flag, Python/library versions, device,
optional dataset-manifest hash, UTC start/end times, wall time, completion
status, data-order seed, checkpoint-selection rule, and whether clean references
were accessed.

The training scripts:

1. enumerate only `train/noisy` and `val/noisy`;
2. derive deterministic crop coordinates from seed/epoch/sample index;
3. seed the shuffled DataLoader order;
4. select `model_best.pth` by minimum noisy-only self-supervised validation
   objective;
5. reserve clean references for explicit evaluation commands;
6. save `model_last.pth`, `model_best.pth`, and `validation.jsonl`.

Deterministic subject manifests omit timestamps and use sorted records. Rician
observations use seeds derived from the base seed, subject ID, and noise label
and are saved once for reuse by every method.

Remaining reproducibility limits:

- CUDA kernels may still be nondeterministic; deterministic-algorithm mode has
  not been enabled or benchmarked.
- The final medical protocol, normalization, foreground mask, and baseline set
  are unresolved.
- HFEN/GMSD require validation against the selected reference implementations;
  `fsim_proxy` is intentionally not called FSIM.
- No dependency lock with hashes is committed yet; minimum versions are
  recorded in package metadata.
