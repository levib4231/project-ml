# Larger noisy-only convergence study

This is an exploratory MPS follow-up at sigma 0.06, seed 42, width 16. It used
all six training subjects and all 358 interior noisy validation slices. Nine
production runs started independently from identical initialization and kept a
fixed final epoch; no hyperparameter or checkpoint was chosen from a score.

| Method | 531 updates | 1062 updates | 2148 updates |
|---|---:|---:|---:|
| Full NGPS | 0.00404150 | 0.00400829 | 0.00397735 |
| Blindspot | 0.00398606 | 0.00394122 | 0.00390822 |
| Adjacent | 0.00399332 | 0.00391652 | 0.00395196 |

The values are masked noisy validation MSE, not clean-image PSNR and not a
clean-risk estimate. The blindspot objective also resembles the monitoring task,
so the table is not a reliable ranking across objectives. No test predictions or
test metrics were produced. A runtime guard on array access confirmed zero
clean/test loads during both the pilot and production.

Full NGPS monitoring MSE fell 1.59% from 531 to 2148 updates. Training loss fell
from 0.0081645 at 531 updates to 0.0056339 in the last epoch of the 2148 run.
That is evidence of numerical convergence, but the monitoring gain is modest and
says nothing yet about denoising quality or an advantage over adjacent training.
The adjacent curve is not monotonic, and no schedule was chosen from it.

All losses and observed gradient norms stayed finite, and every run met its
independently frozen 3600-second MPS cap. Pilot projections were 528.4, 1053.9
and 1962.6 seconds for the three schedules. NGPS wall times were 150.36, 262.09
and 492.64 seconds; production wall times summed to 1648.42 seconds (27.47
minutes), excluding the pilot. No portable MPS peak memory is claimed.

The schedules were 531 = one epoch capped at 531 batches, 1062 = two epochs
capped at 531, and 2148 = two full 1074-batch epochs. The last one expands sample
coverage as well as updates, so it is not a pure continuation of the capped
schedule. All objectives share the matching schedule, and the first-epoch
training records of the 531 and 1062 runs matched exactly for each objective.
Protocol and runtime hashes were verified.

I left the historical T4 and MPS protocols and results untouched. This directory
retains the orchestrator, design, frozen pilot projection, configs, checkpoints,
budget records, access audit and summaries.

The next useful predeclared experiment is a full-workload loss ablation to probe
the dynamic-target and RC contributions. Longer training did not fix the original
unmet criterion, and untouched subjects are still needed before any candidate can
be confirmed independently.
