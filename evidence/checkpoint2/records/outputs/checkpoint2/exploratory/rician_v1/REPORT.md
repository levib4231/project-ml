# BrainWeb/Rician exploratory benchmark

We trained fresh Rician NGPS, blindspot and adjacent models at sigma 0.03, 0.06
and 0.10, nine runs in total. The train 6 / val 2 / test 2 split, references,
normalization, seed 42, shared width 16 and 531-update single-final-epoch
workload match the completed Gaussian MPS benchmark. Test subjects 20 and 44
were scored on complete volumes.

## Frozen test results

| Component sigma | NGPS PSNR | Strongest trained comparator | Comparator PSNR | Gap | Criterion met |
|---|---:|---|---:|---:|---|
| 0.03 | 30.8587 | adjacent | 31.0004 | -0.1417 dB | No |
| 0.06 | 24.6676 | adjacent | 24.2213 | +0.4463 dB | Yes |
| 0.10 | 19.6414 | blindspot | 19.6850 | -0.0436 dB | No |

The original all-sigma success criterion remains unmet. At sigma 0.06 NGPS also
beat NLM PSNR (24.4036 dB), and the gap to the stronger trained comparator was
+0.4441 and +0.4484 dB on subjects 20 and 44 separately. Treat that as a limited
positive signal, not general superiority: the subjects are already exposed, and
the result rests on one seed, two subjects and a single reduced training budget.

Other metrics qualify the PSNR gain. At sigma 0.06 the NGPS/adjacent SSIM was
0.38370/0.37926, but HFEN was 0.99304/0.94775 (lower is better), so NGPS did not
win edge fidelity. Full per-subject and slice metrics, timings, predictions and
figures are in experiment/results/.

## Exact synthetic corruption, different task

Corruption was generated as hypot(x+n1,n2) with independent Gaussian components
of SD sigma and no clipping. These are synthetic Rician samples within float32
precision, not Gaussian-plus-offset approximations. Noise checks passed for
Rayleigh zero-signal moments, nonzero Rice moments against SciPy, determinism and
absence of upper clipping. Reference volumes kept the course /4095 scale with no
per-volume rescaling. At zero signal the positive floor averages 1.253 sigma.
Rician and Gaussian at the same numerical sigma do not have the same effective
corruption MSE. Lower absolute Rician PSNR than Gaussian is therefore neither an
implementation failure nor evidence of an easier or harder task.

We kept the same numerical sigmas for continuity with the course benchmark, not
to mirror the paper's 5/7/9% IXI experiment. BrainWeb, reduced capacity and
training, active-normalized RC and signal scaling still differ from the paper, so
this is not a numerical reproduction. NLM keeps its existing Gaussian-calibrated
parameters and is not a specialized Rician denoiser.

## Safeguards and artifacts

We froze the design before corruption and training, then followed a fresh MPS
pilot with a frozen protocol before production. Every run completed 531 updates
inside the independently declared 3600-second cap. Summed production wall time
was 655.24 seconds, excluding preparation, the pilot and evaluation. The
training/preflight array-access guard recorded zero forbidden loads. Clean
references were used only to generate corruption and to score the frozen
predictions, never for training, tuning, validation or selection. Source,
orchestrator, design and protocol hashes plus all nine completed runs were
verified; 30 subject rows and 30 prediction volumes were retained. Historical
Gaussian outputs are unchanged. Using these results for later decisions cannot
make this test confirmatory, and independent confirmation needs untouched
subjects.
