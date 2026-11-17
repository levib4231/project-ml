# Paired-noise component isolation

This predeclared 8-run study crossed Gaussian/Rician noise with four objectives:
full NGPS, no RC, no dynamic and adjacent. Settings were sigma 0.06, seed 42,
width 16, 531 updates, all six training subjects and 358 interior noisy
validation slices. Every variant was freshly trained with the same schedule and
initialization, and all models were frozen and preflighted before reference
scoring. Test subjects 20 and 44 had already been exposed, so this is exploratory
only: no independent confirmation, no clean-reference tuning and no winner
selection.

## Component-removal results

Mean slice PSNR, averaged equally over the two subjects:

| Objective | Gaussian PSNR | Rician PSNR |
|---|---:|---:|
| Full NGPS | 35.7044 | 24.6676 |
| NGPS without RC | 35.3924 | 24.7772 |
| NGPS without dynamic loss | 36.7034 | 24.2809 |
| Adjacent | 36.0204 | 24.2213 |

Compare within each noise task, not across tasks with different effective
severity.

- Dynamic contribution given RC (full minus no dynamic): -0.9990 dB for
  Gaussian, +0.3866 dB for Rician. The sign flips on both subjects.
- RC contribution given dynamic (full minus no RC): +0.3120 dB for Gaussian,
  -0.1096 dB for Rician. This sign also flips on both subjects.
- Full minus adjacent: -0.3160 dB for Gaussian, +0.4463 dB for Rician,
  reproducing the earlier matched 531 results.

At this seed and budget, the Rician improvement tracks the dynamic-target loss
more than RC. The study does not isolate why that term helps: anatomical
correspondence, target-intensity selection, implicit weighting and optimization
all remain candidates. The design is also not a full RC-by-dynamic factorial;
there is no static-only/no-RC cell.

## Where the Rician gain occurs

Reference-intensity strata and residual statistics were specified before
scoring. They are evaluation-only and never served as training masks, guides or
targets.

Rician full NGPS vs adjacent:

| Reference region | Full mean signed error | Adjacent mean signed error | Full MSE | Adjacent MSE |
|---|---:|---:|---:|---:|
| x<0.02 | +0.06710 | +0.07061 | 0.00459774 | 0.00506043 |
| 0.02<=x<0.10 | +0.03192 | +0.03697 | 0.00136613 | 0.00170319 |
| x>=0.10 | +0.00354 | +0.00883 | 0.00045854 | 0.00051831 |

The low-intensity bin holds about 69.8% of pixels and accounts for about 87.2%
of the volume-MSE reduction of full NGPS vs adjacent. That figure weights
within-subject regional MSE differences by region fractions, then averages the
subjects; it is not a decomposition of the slice-mean PSNR difference.

Full NGPS lowers the overall positive residual mean from 0.05492 to 0.05091
while the overall spatial centered residual variance rises from 0.0007970 to
0.0008519. The MSE gain therefore looks like a reduced positive noise-floor
error rather than simply lower spatial residual variance. This is a
deterministic spatial error decomposition, not an estimator bias/variance study
across repeated noise draws, and it does not prove that NGPS corrects Rician
bias on purpose.

## Structural-quality qualification

Rician HFEN (lower is better): full 0.99304, no RC 1.04389, no dynamic 0.90569,
adjacent 0.94775. The full-NGPS PSNR gain does not imply better edge fidelity.
SSIM: full 0.38370, no RC 0.38320, no dynamic 0.37683, adjacent 0.37926.

Our earlier suggestion that RC helped came from noisy monitoring. Here, removing
RC slightly raises clean PSNR under Rician even though it worsens noisy
monitoring. Noisy monitoring is not a reliable cross-objective clean-risk
ranking. We changed no defaults from these post-test observations.

## Safeguards and execution

All 8 runs completed 531 updates with finite observed losses and gradients, and
all met their independently frozen 3600-second caps. Total production wall time
was 1174.94 seconds (19.58 minutes), excluding pilots and scoring. The runtime
training access guard recorded no clean/test array loads. Shared reference hashes
and splits, source/design/protocol, checkpoint hashes and completed workloads
were checked before scoring, and all 20 subject records and 20 full-volume
predictions were retained.

The first orchestration attempt failed only during summary aggregation (a
subject vs subject_id key) after all training and scoring had finished. The
original execution log, scores, script hash and predictions were preserved.
finalize_report.py aliased the key for reporting only, with no retraining,
rescoring or protocol change. Details are in reporting_recovery.json.

## Bottom line

The observed Rician advantage sits in the dynamic-loss branch and mostly reflects
lower error in low-intensity regions. We have not shown better matching in
general, a broad anatomical advantage or robustness across seeds and noise
levels. Distinguishing matching from target-selection bias needs a separate
frozen guide/target-swap study or repeated seeds, and independent confirmation
needs untouched subjects. Historical benchmarks and default model behavior are
unchanged.
