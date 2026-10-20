# Data Provenance and Licensing Status

## What is verified locally

`Data/Clean` contains ten paired files named
`subject##_t1w_p4.mnc.gz` and `subject##_clean.npy`. Every derived array is
float32 with shape `(181, 256, 256)`. The deterministic inventory records exact
hashes and value statistics in `NGPS/manifests/committed_subject_inventory.json`.

The inventory uses foreground threshold `> 0`. Because every stored voxel is
slightly positive after conversion, the recorded foreground fraction is 1.0 for
all subjects. This is a reproducible file statistic, not an anatomical mask;
the evaluation foreground policy remains unresolved.

`Data/Clean/convert_brain.py` reads each compressed MINC source, obtains a
float32 volume with NiBabel, clips it to 0-4095, divides by 4095, and saves the
derived NumPy array. The derived arrays are therefore normalized to a subset of
`[0,1]`; their observed maxima are about 0.32-0.34.

The embedded history strings inspected in at least one MINC source mention
`mrisim` and anatomical tissue-class inputs. That is consistent with simulated
brain MRI and with the converter’s “BrainWeb” comments, but it is not an
authoritative source or license record.

## What is not verified

- No authoritative download URL, acquisition log, or license text accompanies
  these files.
- The files must not be described as IXI. Their names and dimensions do not
  establish IXI provenance.
- “BrainWeb-derived” is a working hypothesis only, not a report-ready fact.
- Permission to redistribute the raw or derived volumes has not been verified.
- It is unknown whether the project should retain the data in Git, use LFS, or
  move future data to controlled external storage.

## Required resolution record

Before a medical experiment or report claim, add for every source group:

1. authoritative dataset name and release/version;
2. exact download page or instructor distribution record;
3. acquisition date and responsible team member;
4. source license/terms and redistribution status;
5. preprocessing command/version and any orientation changes;
6. verification that source and derived hashes match the committed manifest.

Do not regenerate, move, delete, or re-license the committed files while this
record is unresolved.
