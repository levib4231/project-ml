# Data Format and Isolation

Training volumes are NumPy arrays with shape `(D,H,W)`, a floating dtype, finite
values, and a declared normalization in `[0,1]`. The loader rejects invalid
values; it does not silently clip them.

```text
DATA_ROOT/
  train/noisy/*.npy
  val/noisy/*.npy
```

Training inventory objects contain only a case ID and noisy path. A separate,
explicit evaluation inventory may attach matching clean references. Clean
volumes must not be used to train, build guides, select checkpoints, or tune
hyperparameters unless a later approved protocol says otherwise.

Random crops are derived from the configured seed, epoch, and sample index. The
dataset caches at most one raw/guide volume pair per worker by default to bound
memory use. Palmetto worker count and memory still require measurement before a
full run.

The files under `Data/Clean` are committed project artifacts whose source and
license are unresolved. See [DATA_PROVENANCE.md](DATA_PROVENANCE.md). Do not
identify them as IXI, redistribute them, or regenerate/delete them based only
on filenames or converter comments.
