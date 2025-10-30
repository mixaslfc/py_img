import numpy as np
from skimage import io, color

# Φόρτωση μιας έγχρωμης εικόνας εκπαίδευσης (σε μορφή RGB)
# Υποθέτουμε ότι η εικόνα είναι σε float (π.χ. 0.0 έως 1.0)
image_rgb = io.imread('data/training_image.jpg').astype(np.float64) / 255.0

# Μετατροπή από RGB σε Lab
image_lab = color.rgb2lab(image_rgb)

# Διαχωρισμός των καναλιών
L_channel = image_lab[:, :, 0]
a_channel = image_lab[:, :, 1]
b_channel = image_lab[:, :, 2]

print(f"Διαστάσεις L: {L_channel.shape}")
print(f"Εύρος τιμών L: {L_channel.min():.2f} έως {L_channel.max():.2f}")
print(f"Εύρος τιμών a: {a_channel.min():.2f} έως {a_channel.max():.2f}")
print(f"Εύρος τιμών b: {b_channel.min():.2f} έως {b_channel.max():.2f}")