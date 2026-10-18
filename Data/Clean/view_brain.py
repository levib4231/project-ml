from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

folder = Path(__file__).resolve().parent
volume = np.load(folder / "subject05_clean.npy")

middle_slice = volume[volume.shape[0] // 2]

plt.imshow(middle_slice, cmap="gray")
plt.title("Subject 05 - middle slice")
plt.axis("off")
plt.show()