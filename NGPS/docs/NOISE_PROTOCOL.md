# Noise Simulation Protocol

## Rician module

`datasets/noise.py` implements magnitude-MRI corruption:

```text
y = sqrt((x + n_real)^2 + n_imag^2)
n_real, n_imag ~ Normal(0, sigma^2), independently
```

The caller must choose one of two explicit sigma references:

- `absolute`: sigma is already expressed in normalized intensity units.
- `data_range_fraction`: effective sigma is `sigma * (data_max - data_min)`.

Output clipping is off by default because clipping changes the distribution.
`--clip-output` must be supplied explicitly when a `[0,1]` training loader is
required. A deterministic seed is derived from the base seed, subject ID, and
noise label, so every compared method can consume the same stored observation.

## Unresolved 5%/7%/9% convention

The local NGPS paper labels IXI corruption as 5%/7%/9% Rician noise, but the
available paper text does not fully define the percentage relative to intensity
normalization. This repository does not assign a meaning to those labels.
Choosing `sigma=0.05` with `data_range_fraction` would be an explicit new
experiment convention, not a verified reproduction of the paper.

The current checkpoint brief instead asks for additive Gaussian sigma
0.03/0.06/0.10 on `[0,1]`. That simulator is not added here because the
readiness plan prioritized a Rician foundation and the final protocol remains a
project decision.
