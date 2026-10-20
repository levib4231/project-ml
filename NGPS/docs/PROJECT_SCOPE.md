# Project Scope and Decision Log

## Repository identity

- Observed configured remote: `https://github.com/levib4231/project-ml.git`.
- Final long-term owner/account: unresolved.
- Scope: course-scale adaptation and partial reproduction foundation, not a
  complete paper reproduction.

## Evidence hierarchy

1. Repository code, generated manifests, test output, and run artifacts support
   implementation claims.
2. `T3_Checkpoint_1_Background_and_Model_Selection.pdf` defines the current
   course checkpoint proposal: public BrainWeb or an instructor-released
   alternative, roughly 6/2/2 subjects, and Gaussian sigma
   0.03/0.06/0.10 on `[0,1]`.
3. `4921.pdf` describes the NGPS paper experiment, including IXI-T1 and Rician
   noise levels labeled 5%/7%/9%.
4. Filenames, comments, and embedded file history are clues, not authoritative
   provenance.

Where these sources conflict, the repository keeps the implementation generic
and records the missing decision rather than silently choosing one.

## Implemented versus planned

| Component | Status | Permitted claim |
|---|---|---|
| BF+MF guide, dynamic mask, local patch matching, Top-K raw targets | Implemented and smoke-tested | Project implementation exists |
| NAFNet training and inference | Implemented and smoke-tested | Synthetic pipeline only until medical runs exist |
| Manifest and subject split tools | Implemented and unit-tested | Deterministic and subject-disjoint |
| Rician simulator | Implemented and unit-tested | Equation and selected sigma reference are reproducible |
| Noisy-only validation and run manifests | Implemented | Clean references are not opened by training inventory/datasets |
| Subject-first metric aggregation | Implemented and unit-tested | Project-local metric outputs, with FSIM proxy clearly named |
| Existing NS-N2N code | Implemented as a project-local comparator | “NS-N2N-style”; no external-equivalence claim |
| NLM, blind-spot, naïve neighbor | Not implemented | Planned only |
| Confidence gating | Not implemented | Planned only |
| Medical benchmark results | Not run | No numerical or clinical claim |

## Decisions still required

1. Verify dataset source URLs, license/terms, and whether every committed pair
   comes from the same source.
2. Decide the course protocol: checkpoint Gaussian/6-2-2, paper-like Rician, or
   another instructor-approved design.
3. If Rician is chosen, define what a percentage means: absolute sigma after a
   stated normalization, fraction of fixed data range, or another cited rule.
4. Confirm required classical and learned baselines before implementation.
5. Confirm validation selection, foreground-mask policy, metric references, and
   compute target.
6. Decide storage strategy without rewriting history or deleting committed data.
