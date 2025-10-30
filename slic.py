# slic.py
# -------
# Τρέχει SLIC για όλες τις training εικόνες και αποθηκεύει τα segments.
# Επίσης κρατάει το πρώτο ως "segments" για συμβατότητα.

import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.segmentation import slic, mark_boundaries
from import_image import training_rgbs, training_labs

N_SUPERPIXELS = 400
COMPACTNESS = 10

segments_list = []
for idx, lab in enumerate(training_labs):
    seg = slic(
        lab,
        n_segments=N_SUPERPIXELS,
        compactness=COMPACTNESS,
        channel_axis=-1,
        start_label=1,
    )
    segments_list.append(seg)
    print(f"[slic] Εικόνα {idx}: δημιουργήθηκαν {seg.max()} superpixels.")

    # Προαιρετική οπτικοποίηση για την 1η εικόνα
    if idx == 0:
        marked = mark_boundaries(training_rgbs[0], seg)
        plt.imshow(marked)
        plt.title("Superpixels (1η εικόνα εκπαίδευσης)")
        plt.axis("off")
        plt.show()

# Αποθήκευση όλων
np.savez("artifacts/slic_segments.npz", *segments_list)

# Συμβατότητα
segments = segments_list[0]

print("ΤΕΛΟΣ: αποθηκεύτηκαν τα segments για όλες τις εικόνες.")
