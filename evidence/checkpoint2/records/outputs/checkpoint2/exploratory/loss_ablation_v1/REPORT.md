# Full-workload loss ablation

Four independently initialized matched runs on MPS: sigma 0.06, seed 42, width
16, 2148 updates each (two full 1074-batch epochs), all six training subjects and
all 358 interior noisy validation slices. We froze the design and pilot
projections before production. The runtime access guard recorded no clean or test
array loads.

| Variant | Final noisy masked validation MSE | Difference vs full NGPS | Production wall seconds |
|---|---:|---:|---:|
| Full NGPS | 0.0039773530 | reference | 326.66 |
| NGPS without RC | 0.0040286779 | +1.290% | 323.11 |
| NGPS without dynamic loss | 0.0039554098 | -0.552% | 327.28 |
| Adjacent | 0.0039519615 | -0.638% | 131.73 |

These are noisy-only monitoring scores. They are not clean-image quality, not
clean-risk estimates and not a reliable ranking of objectives. All variants are
reported; none was treated as a winner or used to replace the default NGPS
implementation.

## Interpretation

Removing RC raises this monitoring error. Removing the dynamic term lowers it
slightly. That points at the dynamic target as a mechanism worth investigating,
but it does not prove that retrieval hurts clean denoising or that RC helps it.
Without-dynamic monitoring rose slightly between epochs (0.0039522943 to
0.0039554098), and we kept the fixed final epoch without post-hoc early stopping
or schedule changes.

The dynamic region covered about 1.296% of training pixels on average, yet its
separately normalized loss in full NGPS was 0.0014542 against 0.0040147 for the
static region at epoch 2. A small area does not imply a small optimization
influence. This on its own does not identify bad matches or justify changing loss
normalization inside the frozen protocol.

All observed losses and gradient norms were finite, and every run met its
independently frozen 3600-second MPS cap. Production wall times sum to 1108.79
seconds (18.48 minutes), excluding pilots. Pilot projections were 1555.8, 1252.6,
1186.5 and 231.8 seconds. Ablation paths still compute matching and neighbor
forwards, so their timings are not optimized savings. Hardware startup and cache
conditions differed from earlier studies; cross-study wall-time differences are
not algorithmic improvements.

Full NGPS and adjacent reproduced their corresponding convergence 2148 runs
exactly. Design, protocol and runtime source hashes were verified, and the
historical medical benchmarks remain untouched.

## Next controlled follow-up

Before claiming better quality, repeat the predeclared ablations at more seeds
and noise levels using noisy-only development, and investigate target reliability
on training observations. Confidence weighting or loss renormalization must
arrive as a separately named, tested follow-up, not as a quiet default change.
Independent quality confirmation needs untouched subjects and a frozen
evaluation protocol.
