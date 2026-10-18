from pathlib import Path
import gzip
import shutil
import tempfile

import nibabel as nib
import numpy as np

folder = Path(__file__).resolve().parent
files = sorted(folder.glob("subject*_t1w_p4.mnc.gz"))

if not files:
    print("No matching .mnc.gz files found in", folder)

for source in files:
    output = folder / source.name.replace("_t1w_p4.mnc.gz", "_clean.npy")

    if output.exists():
        print("Already exists, skipping:", output.name)
        continue

    # Unzip temporarily so NiBabel can read the image.
    with tempfile.TemporaryDirectory() as temp_dir:
        uncompressed = Path(temp_dir) / "brain.mnc"

        with gzip.open(source, "rb") as zipped:
            with uncompressed.open("wb") as temp_file:
                shutil.copyfileobj(zipped, temp_file)

        image = nib.load(str(uncompressed))
        volume = image.get_fdata(dtype=np.float32)

    # Scale BrainWeb's 0–4095 intensity values to 0–1.
    volume = np.clip(volume, 0, 4095) / 4095.0
    np.save(output, volume.astype(np.float32))

    print("Saved:", output.name, "| shape:", volume.shape)