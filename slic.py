# slic.py
# -------
# Τρέχει SLIC για όλες τις training εικόνες και αποθηκεύει τα segments.
# Επίσης κρατάει το πρώτο ως "segments" για συμβατότητα.

import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.segmentation import slic, mark_boundaries
from import_image import training_rgbs, training_labs

N_SUPERPIXELS = 200
COMPACTNESS = 10

output_image_dir = "artifacts/images"
os.makedirs(output_image_dir, exist_ok=True)

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
    print(f"Εικόνα {idx}: δημιουργήθηκαν {seg.max()} superpixels.")

    # Αποθήκευση εικόνας με όρια superpixels
    marked = mark_boundaries(training_rgbs[idx], seg)
    
    # Χρησιμοποιούμε τη μεταβλητή που ορίσαμε πάνω
    out_path = f"{output_image_dir}/slic_image_{idx}.png" 
    
    plt.imsave(out_path, marked)
    print(f"Αποθηκεύτηκε το {out_path}")
    
    # Αυτό το κομμάτι είναι σωστό για να βλέπεις κάθε εικόνα
    plt.imshow(marked)
    plt.title(f"Superpixels (εικόνα {idx + 1})")
    plt.axis("off")
    plt.show()
    plt.close()


# Αποθήκευση όλων
np.savez("artifacts/slic_segments.npz", *segments_list)

# Συμβατότητα
segments = segments_list[0]

print("ΤΕΛΟΣ iii: αποθηκεύτηκαν τα segments για όλες τις εικόνες.")
