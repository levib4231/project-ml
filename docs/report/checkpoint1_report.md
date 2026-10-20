# Medical Image Denoising: Checkpoint 1 Evidence Draft

> Status: evidence-grounded scaffold, not submission-ready. Group members,
> instructor decisions, verified data provenance, final protocol, and results
> remain incomplete.

**Course:** CPSC 4430/6430, Fall 2026<br>
**Group members:** TODO - not present in repository evidence<br>
**Draft date:** 2026-10-20

## Project summary

This project studies noisy-only denoising of volumetric MRI using
Neighbor-Guided Patch Sampling (NGPS). The current repository implements the
NGPS training-time supervision path and a reproducible synthetic pipeline. It
does not yet establish medical benchmark performance or clinical utility.

## Original NGPS contribution

The local NGPS paper frames adjacent-slice anatomical mismatch as displaced
supervision. It filters slices to form structural guides, detects discrepant
regions, searches a local neighboring-slice window for similar guide patches,
and retrieves raw noisy values at Top-K matched coordinates. Static regions use
same-coordinate neighboring targets. A NAFNet denoiser is used at inference;
guide generation and patch search are training-only operations.

The paper reports CT experiments and an IXI-T1 experiment with synthetic
Rician levels labeled 5%/7%/9%. The percentage convention is not sufficiently
specified in the local paper text to reproduce without an additional decision.

## Course adaptation and research question

The current checkpoint brief proposes public BrainWeb data or an
instructor-released volumetric alternative, roughly 6/2/2 subjects, Gaussian
sigma 0.03/0.06/0.10 on `[0,1]`, and noisy-only validation or a fixed schedule.
This differs from the paper’s IXI/Rician experiment.

**Provisional research question:** Under one instructor-approved corruption
protocol and subject-disjoint split, does NGPS preserve structure better than
the required matched baselines while training without clean targets?

This question is provisional until the dataset, noise protocol, baselines,
metric references, and compute budget are approved.

## Dataset and provenance

The repository contains ten paired compressed MINC sources and float32 NumPy
volumes of shape `(181,256,256)`. A committed inventory records hashes and
statistics. Converter comments and embedded MINC history are consistent with
simulated brain MRI, but no authoritative source URL or license record is
present. The files must not be called IXI; “BrainWeb-derived” remains an
unverified hypothesis.

## Subject split and leakage controls

A deterministic draft split uses seed 42 and explicit counts 6/2/2. It is a
planning artifact, not an approved final split. All slices and crops from a
subject remain in one split.

Training inventories contain noisy paths only. Training and checkpoint
selection enumerate `train/noisy` and `val/noisy`; clean references are opened
only by explicit evaluation commands. Each run records whether clean references
were accessed.

## Noise protocol

The repository includes a configurable Rician simulator:

```text
y = sqrt((x+n_real)^2 + n_imag^2)
```

The caller must choose absolute sigma or sigma as a fraction of a fixed data
range, plus an explicit clipping policy. No paper-equivalent 5%/7%/9% claim is
made. The checkpoint’s requested Gaussian protocol is not yet implemented;
the final choice is an instructor/team decision.

## Implementation mapping

| Concept | Repository implementation |
|---|---|
| Filtered guide | `NGPS/ngps/guide.py` |
| Static/changing mask | `make_misalignment_mask` |
| Local patch matching and Top-K raw retrieval | `NGPS/ngps/sampler.py` |
| Hybrid and regional losses | `NGPS/ngps/losses.py` |
| Denoiser | `NGPS/ngps/nafnet.py` |
| Noisy-only validation and metadata | training scripts and `NGPS/ngps/run_metadata.py` |

## Baselines and proposed improvement

The repository has a project-local NS-N2N-style comparator, but it has not been
validated as an exact reproduction. NLM, a masked-pixel blind-spot model, and an
unmasked same-coordinate adjacent-slice baseline are required by the current
checkpoint and remain unimplemented. Confidence gating is also unimplemented.
No baseline or gating result is claimed.

## Evaluation plan

Evaluation computes 2D slice metrics, averages them within subject, and then
summarizes subject means with mean, sample standard deviation, and a 95%
Student-t interval. The current default is full-image evaluation. PSNR and SSIM
use data range 1. HFEN and GMSD still need validation against approved
references. The current feature-similarity metric is named `fsim_proxy` and is
not reference FSIM.

## Verified practical-start evidence

- Compilation and NGPS tensor smoke tests run on CPU.
- Unit tests cover manifests, splits, isolation, Rician simulation, sampler
  boundaries, config validation, aggregation, and run metadata.
- A tiny non-medical CPU run completed generation, one NGPS optimization batch,
  noisy-only validation, checkpoint selection, inference, and evaluation.
- These checks validate software plumbing only and are not scientific results.

## Limitations and next decisions

1. Verify dataset source and redistribution terms.
2. Approve Gaussian versus Rician corruption and exact normalization.
3. Approve baseline definitions and a common compute budget.
4. Validate metric implementations and decide foreground masking.
5. Confirm Palmetto hardware/software and run deterministic medical pilots.
6. Implement required baselines before any comparative claim.

## References

- Cho, Jaehyun, and YoungJoon Yoo. *NGPS: Structure-Preserving
  Self-Supervised Denoising via Neighbor-Guided Patch Sampling*. Local 2026
  preprint; final venue metadata must be verified.
- Additional dataset, metric, and baseline references: TODO after protocol
  approval and provenance verification.
