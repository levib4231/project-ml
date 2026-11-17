# Checkpoint implementation follow-ups

Inspected October 23, 2026, against
`5ae9934f027e3ceb735c7a6135f428495c1960a2` on GitHub main. This revision changes
documents and presentation assets, not experiment code or data.

## Open scientific and reporting issues

1. Regional consistency scale. `NGPS/ngps/losses.py` normalizes RC by active
   pixels, while NGPS paper Eq. 9 uses an unnormalized masked squared L2 sum. The
   same lambda therefore has a different strength in each. Record the
   implementation choice and evaluate a convention separately before claiming
   numerical reproduction. Reconstruction also uses a denominator clamp in code
   where the paper's Eqs. 7–8 use an additive epsilon.
2. Medical provenance and normalization. Source identities, hashes and the agreed
   split are committed in `NGPS/configs/subjects_v1.json`, but the intensity-scale
   interpretation and acquisition/conversion provenance still need verification.
   The current preparer implements explicit normalization and unclipped Gaussian
   noise. Synthetic evidence is not a reason to claim medical runs.
3. Confidence weighting. Matching diagnostics exist, but a confidence-weighted
   dynamic loss is still a proposed course extension. Fix its rule before test
   evaluation, then run the matched ablations.
4. Metric convention. The current 7×7 uniform SSIM window differs from the
   original SSIM paper's Gaussian window. HFEN uses SciPy LoG at sigma 1.5;
   validate the intended kernel convention. Keep the gradient metric named
   `SobelGradientMagnitudeSimilarity`; it is not FSIM.
5. Runtime scope. `inference_seconds` covers complete-volume prediction,
   transfers and assembly, excludes I/O and model loading, and has no warmup. The
   planned course load-to-save `T_volume` needs a separate synchronized
   end-to-end measurement. Do not treat the two scopes as the same.
6. Medical evidence and budget. The repository has a synthetic evidence log, not
   a verified medical benchmark. Measure the planned medical pilot and fix
   matched workloads below ten GPU minutes before claiming that budget is met.

## Local checkout drift

The older local checkout inspected during planning has an empty `Data/Noisy`, a
different historical random split, folder-based training, a Rician-only medical
simulator and missing baseline objectives. Newer GitHub main already provides the
agreed split, unclipped Gaussian preparation, NLM, blind-spot and adjacent
objectives, fixed final-epoch selection and manifest-based evaluation. Those
older findings are not unresolved defects in current GitHub main.

The root README targets the inspected current GitHub version. Update the older
local checkout separately before running its commands. Its code and data are
otherwise unchanged.
