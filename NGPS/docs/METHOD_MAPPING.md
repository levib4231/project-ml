# Method mapping and reference comparison

| Component | Implementation |
|---|---|
| Subject splits, seeds, path leakage | `datasets/manifest.py`, `configs/subjects_v1.json` |
| Explicit normalization, unclipped Gaussian noise | `datasets/preparation.py`, `scripts/prepare_data.py` |
| Float BF+MF guide, direction-aware mask | `ngps/guide.py` |
| Guide patch SSD, Top-K raw center retrieval | `ngps/sampler.py` |
| Static/dynamic reconstruction + regional consistency | `NGPSHybridLoss` in `ngps/losses.py` |
| NS-N2N-style masked adjacent reconstruction + RC/IC | `NSN2NLoss` in `ngps/losses.py` |
| Unmasked same-coordinate adjacent target baseline | `adjacent_loss` in `ngps/baselines.py` |
| Masked-pixel blind-spot training baseline | `mask_pixels`, `blindspot_loss` in `ngps/baselines.py` |
| Actual slice-wise non-local means | `nlm_volume` in `ngps/baselines.py` |
| Shared training / fixed selection / noisy validation | `ngps/training.py` |
| Held-out subject metrics and timing | `ngps/evaluation.py`, `scripts/evaluate_manifest.py` |
| Denoiser | `ngps/nafnet.py` |

The sampler edge-replicates both guide images before forming the shifted patch
SSD, so there is no zero-cost padding. Raw retrieval uses the same
edge-replicated coordinate convention. Images smaller than the search radius are
safe. At borders several shifts can map to the same pixel, and Top-K does not
enforce unique physical pixels. Costs are returned for both directions. The
guide, search and masks are training-only and play no part in neural inference.

The blind-spot baseline replaces Bernoulli-5% pixels (at least one per image)
with another noisy pixel from a randomly chosen nonzero horizontal cyclic
offset. Targets are original noisy pixels and the loss is evaluated only at
replacements. It uses no clean or adjacent target. This is a concrete
masked-pixel baseline, not a claim of exact Noise2Void reproduction or a strictly
J-invariant network. At inference it uses the ordinary unmasked NAFNet forward
pass. NLM is 2D scikit-image fast NLM with patch size 5, search distance 6 and
`h=0.8*sigma`, using the known synthetic sigma. These defaults are executable
values, not a tuning recommendation.

## NAFNet comparison

The locally available reference source was
[megvii-research/NAFNet, `basicsr/models/archs/NAFNet_arch.py`](https://github.com/megvii-research/NAFNet/blob/main/basicsr/models/archs/NAFNet_arch.py).
The inspected file's SHA-256 was
`01b22270cc93f1bb90c0e3e4490e98b023fcf73f8552860b4a9ee880ce5c6967`.
Its upstream commit was not established here because the URL is not an immutable
pin.

- The reference `forward` ends with `ending(x) + padded_input`. The earlier
  local implementation left out that global residual; it is restored now.
- The reference uses zero padding for stride padding. The local reflect padding
  now matches it and also avoids failures on very small images.
- SimpleGate, depthwise expansion, simplified channel attention, encoder skip
  additions, pixel-shuffle upsampling and zero-initialized beta/gamma follow the
  reference. We use native `nn.LayerNorm` over channels instead of the custom
  reference autograd LayerNorm and import no reference weights.
- Dropout is absent, which is equivalent to the reference default rate of 0.
  This is plain NAFNet, not the reference `NAFNetLocal` pool-conversion variant.
- The application uses one input channel and config-specified block counts; it
  does not claim identical architecture sizes to every published experiment.

New checkpoints record `architecture_version=2`. Inference rejects legacy
checkpoints without that tag, because identical state-dict shapes would
otherwise hide the changed forward semantics. Tests zero the ending head and
verify exact identity through the global residual, including odd and tiny
spatial shapes. Reference weight parity and training-result reproduction remain
unverified.
