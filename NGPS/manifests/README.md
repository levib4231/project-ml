# Manifests

- `committed_subject_inventory.json`: deterministic file facts, hashes, and
  statistics. Dataset identity remains explicitly unverified. Its `> 0`
  foreground threshold includes every stored voxel and must not be treated as
  an anatomical foreground mask.
- `committed_subject_split_seed42.json`: provisional 6/2/2 subject-level split
  derived from the inventory with seed 42. It follows the current checkpoint’s
  rough count only; it is not an approved final experimental split.

Regenerate and compare:

```bash
python NGPS/scripts/make_manifest.py \
  --data-dir Data/Clean \
  --dataset-name committed-subject-volumes-identity-unverified \
  --preprocessing-version convert_brain.py-v1 \
  --provenance-note "identity and license unresolved" \
  --output /tmp/committed_subject_inventory.json

python NGPS/scripts/make_split_manifest.py \
  --manifest /tmp/committed_subject_inventory.json \
  --counts 6 2 2 --seed 42 \
  --output /tmp/committed_subject_split_seed42.json
```

The generator intentionally emits no timestamp so identical inputs and options
produce byte-identical JSON.
