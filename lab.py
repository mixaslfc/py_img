import os
import numpy as np
import joblib
from sklearn.cluster import MiniBatchKMeans
from import_image import training_labs

os.makedirs("artifacts", exist_ok=True)

N_COLOR_CLASSES = 16
MAX_PER_IMAGE = 50000  # περιορισμός δείγματος γιατί είδαμε οτι μερικές εικόνες έχουν πάρα πολλά pixels

all_ab = []
for lab in training_labs:
    ab = lab[:, :, 1:3].reshape(-1, 2)
    if ab.shape[0] > MAX_PER_IMAGE:
        idx = np.random.choice(ab.shape[0], MAX_PER_IMAGE, replace=False)
        ab = ab[idx]
    all_ab.append(ab)

all_ab = np.vstack(all_ab)
print(f"Συνολικά (a,b) δείγματα για k-means: {all_ab.shape[0]}")

kmeans = MiniBatchKMeans(
    n_clusters=N_COLOR_CLASSES,
    random_state=42,
    n_init=5,
    batch_size=4096,
)
kmeans.fit(all_ab)

joblib.dump(kmeans, "artifacts/kmeans_palette.joblib")

color_palette = kmeans.cluster_centers_
print(f"Παλέτα (πρώτα 5):\n{color_palette[:5]}")


quantized_labels_list = []
for idx, lab in enumerate(training_labs):
    h, w, _ = lab.shape
    ab_full = lab[:, :, 1:3].reshape(-1, 2)
    labels = kmeans.predict(ab_full).reshape(h, w)
    quantized_labels_list.append(labels)
    print(f"Εικόνα {idx}: quantized_labels shape = {labels.shape}")

np.savez("artifacts/quantized_labels.npz", *quantized_labels_list)
