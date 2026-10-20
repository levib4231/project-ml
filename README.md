# Project ML: Noisy-Only Volumetric MRI Denoising

This repository is a course-scale adaptation of Neighbor-Guided Patch Sampling
(NGPS). It currently provides a tested synthetic pipeline and reproducibility
infrastructure for a future subject-level medical-imaging experiment. It does
**not** yet provide medical benchmark results or a complete reproduction of the
paper.

The configured Git remote is `https://github.com/levib4231/project-ml.git`.
Whether that account is the final long-term owner remains an open project
decision.

## Current claim boundary

| Area | Repository status |
|---|---|
| NGPS target construction and NAFNet training | Implemented; synthetic smoke coverage only |
| Committed medical volumes | Inventoried and hashed; dataset identity and license are not verified |
| Subject splits | Deterministic tooling plus a provisional 6/2/2 draft split |
| Rician corruption | Configurable and tested; no 5%/7%/9% convention has been adopted |
| Train/validation isolation | Noisy-only loaders, noisy-only checkpoint selection, and run manifests implemented |
| Evaluation | Slice metrics aggregated within subject, then across subjects |
| FSIM | `fsim_proxy` only; not reference or paper-equivalent FSIM |
| NS-N2N | Project-local “NS-N2N-style” comparator; not a validated reproduction |
| NLM, masked blind-spot, naïve-neighbor baselines | Planned, not implemented |
| Confidence-gated NGPS | Planned, not implemented |
| Medical benchmark and report results | Not run; no results are claimed |

The current course checkpoint and the local NGPS paper describe different
candidate protocols. The checkpoint asks for public BrainWeb (or an
instructor-released alternative), roughly 6/2/2 subjects, and Gaussian noise at
0.03/0.06/0.10 on `[0,1]`. The paper evaluates IXI-T1 with Rician levels labeled
5%/7%/9%, but does not define enough of that percentage convention in the local
paper to reproduce it safely. See [project scope](NGPS/docs/PROJECT_SCOPE.md)
and [data provenance](NGPS/docs/DATA_PROVENANCE.md).

## Repository layout

```text
Data/Clean/       committed MINC sources and derived NumPy volumes (do not delete)
NGPS/             model, data, training, evaluation, docs, and tests
docs/report/      evidence-grounded report source when present
4921.pdf          local NGPS paper
T3_*.pdf          current course checkpoint brief
```

## Quick verification

From the repository root:

```bash
python -m pip install -r NGPS/requirements.txt
python -m pip install -r NGPS/requirements-dev.txt
python -m compileall -q NGPS Data/Clean
python NGPS/scripts/smoke_test.py
pytest -q
```

The synthetic demo is non-medical and exists only to verify the software path:

```bash
python NGPS/scripts/make_synthetic_demo_data.py --output-root /tmp/project-ml-synthetic
python NGPS/scripts/train_ngps.py \
  --data-root /tmp/project-ml-synthetic \
  --config NGPS/configs/synthetic_demo.yaml \
  --output-dir /tmp/project-ml-run \
  --device cpu
```

Training reads only `SPLIT/noisy/*.npy`. Clean references are attached only by
explicit evaluation commands.

## Data and licensing

The ten committed source/derived volume pairs are retained as required. Local
file evidence is consistent with MRI simulation tooling, but there is no
authoritative acquisition URL or license record in the repository. They must
not be described as IXI, and “BrainWeb-derived” remains a hypothesis until
provenance is verified.

The NGPS package code carries the nested [MIT license](NGPS/LICENSE). No root
license is added because applying that license to the medical data, PDFs, and
other project assets would be unjustified while their provenance and terms are
unresolved.
