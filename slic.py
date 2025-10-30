from skimage.segmentation import slic, mark_boundaries
import matplotlib.pyplot as plt
import numpy as np
from import_image import image_rgb, image_lab

# Παράμετροι SLIC
N_SUPERPIXELS = 400
COMPACTNESS = 10 # (Υψηλότερο = πιο τετραγωνισμένα superpixels)

# Εφαρμογή του SLIC
segments = slic(image_lab, 
                n_segments=N_SUPERPIXELS, 
                compactness=COMPACTNESS, 
                channel_axis=-1, # Το Lab έχει 3 κανάλια
                start_label=1)

print(f"Δημιουργήθηκαν {np.max(segments)} superpixels.")

# Οπτικοποίηση 
marked_image = mark_boundaries(image_rgb, segments)
plt.imshow(marked_image)
plt.title("Superpixels (SLIC)")
plt.show()