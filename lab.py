from sklearn.cluster import MiniBatchKMeans
import numpy as np
import joblib
from import_image import image_lab

ab_channels = image_lab[:, :, 1:3] 
h, w, _ = ab_channels.shape

# Φέρνουμε όλα τα (a,b) pixels σε μια 2D λίστα [N_pixels, 2]
pixel_data_ab = ab_channels.reshape((h * w, 2))

# Ορίζουμε τον αριθμό των "κάδων" χρώματος (π.χ., 32 ή 64)
N_COLOR_CLASSES = 32


# Εκπαιδεύουμε το K-Means
kmeans = MiniBatchKMeans(n_clusters=N_COLOR_CLASSES, random_state=42, n_init=3)
kmeans.fit(pixel_data_ab)

# Αυτή είναι η 'παλέτα' μας
color_palette = kmeans.cluster_centers_
print(f"Η παλέτα μας (Κεντροειδή K-Means):\n {color_palette[:5]}...") # 5 πρώτα χρώματα

# Τώρα μπορούμε να πάρουμε τις ετικέτες (0-31) για κάθε pixel
quantized_labels = kmeans.labels_.reshape((h, w))
print(f"Διαστάσεις κβαντισμένων ετικετών: {quantized_labels.shape}")

# Αποθηκεύουμε το μοντέλο K-Means για μελλοντική χρήση
joblib.dump(kmeans, 'kmeans_palette.joblib')
print("Το μοντέλο K-Means (παλέτα) αποθηκεύτηκε στο 'kmeans_palette.joblib'")