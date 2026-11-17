# Exploratory MPS comparison pilot

Ten runs on subject04 at sigma 0.06, with matched 32/64-update schedules, width
16, batch 1, seed 42 and fixed-final-checkpoint selection. Every run stayed
within the independently declared 600-second local pilot guard. Summed run time
was 112.75 seconds; individual times include startup and diagnostic overhead and
are not controlled performance comparisons.

| Variant | Noisy masked validation MSE: 32 updates | 64 updates |
|---|---:|---:|
| Full NGPS | 0.01214749 | 0.01020065 |
| Blindspot | 0.01178979 | 0.00942470 |
| Adjacent | 0.01238906 | 0.01002941 |
| NGPS without RC | 0.01257589 | 0.01089283 |
| NGPS without dynamic term | 0.01162973 | 0.00897701 |

These scores are noisy-only monitoring, not clean-risk estimates. Blindspot
training in particular resembles this validation task. Do not use the table to
pick a winner, checkpoint or hyperparameter. The pilot computed no clean
PSNR/SSIM and loaded no test arrays.

## Observations

- Every variant's noisy monitoring MSE fell between 32 and 64 updates.
- No nonfinite loss or gradient norm interrupted training.
- Full NGPS mean training loss fell from 0.0198651 to 0.0150573.
- Dynamic regions averaged 0.5435% of training pixels, but active-pixel
  normalization gives their reconstruction a separate mean: at epoch 2 the
  static loss was 0.0071594 and the dynamic loss 0.0046345. A small mask area
  does not make the dynamic contribution negligible.
- On the single fixed training diagnostic crop, dynamic masks covered about
  0.82% per direction, with a mean selected offset of about 4.58 pixels. That
  alone cannot show whether matches were anatomically correct.
- The independently restarted 64-update runs matched their 32-update
  counterparts' first-epoch training records exactly. Protocol and source
  hashes verified.

## Verification

- 42 unit tests passed, including guards against clean/test array access, exact
  CPU update equivalence with optional diagnostics, and ablation arithmetic.
- Compile and diff checks passed.
- The existing synthetic nine-run Checkpoint 2 benchmark still passed.

## Next step

Run the predeclared matched full-training convergence study before considering
capacity or confidence weighting. This feasibility pilot does not show an
improvement over adjacent supervision. Keep the historical benchmark findings
as they are. More test scoring on already exposed subjects stays exploratory;
independent confirmation needs untouched subjects.
