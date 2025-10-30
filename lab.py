# lab.py
# ------
# Παίρνει τα (a,b) από τις εικόνες εκπαίδευσης και κάνει ένα κοινό K-Means.
# Αποθηκεύει:
#   artifacts/kmeans_palette.joblib
#   artifacts/quantized_labels.npz   (ένα 2D label-map για κάθε training εικόνα)

import os
import numpy as np
import joblib
from sklearn.cluster import MiniBatchKMeans
from import_image import training_labs

os.makedirs("artifacts", exist_ok=True)

# Συλλογή ΟΛΩΝ των (a,b) pixels
all_ab = []
for lab in training_labs:
    ab = lab[:, :, 1:3].reshape(-1, 2)
    all_ab.append(ab)
all_ab = np.vstack(all_ab)

N_COLOR_CLASSES = 32

print(f"Συνολικά (a,b) δείγματα για k-means: {all_ab.shape[0]}")

kmeans = MiniBatchKMeans(
    n_clusters=N_COLOR_CLASSES,
    random_state=42,
    n_init=5,
    batch_size=4096,
)
kmeans.fit(all_ab)

color_palette = kmeans.cluster_centers_
print(f"Παλέτα (πρώτα 5):\n{color_palette[:5]}")

# Για ΚΑΘΕ training εικόνα υπολογίζουμε τον χάρτη ετικετών
quantized_labels_list = []
for idx, lab in enumerate(training_labs):
    h, w, _ = lab.shape
    ab = lab[:, :, 1:3].reshape(-1, 2)
    labels = kmeans.predict(ab).reshape(h, w)
    quantized_labels_list.append(labels)
    print(f"Εικόνα {idx}: quantized_labels shape = {labels.shape}")

# Αποθήκευση
joblib.dump(kmeans, "artifacts/kmeans_palette.joblib")
np.savez("artifacts/quantized_labels.npz", *quantized_labels_list)

quantized_labels = quantized_labels_list[0]
quantizeds_labels_list = quantized_labels_list

print("ΤΕΛΟΣ: αποθηκεύτηκε η παλέτα και τα quantized labels.")
