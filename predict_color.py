# predict_color.py
# ----------------
# Βήμα vi: παίρνουμε ΜΙΑ ασπρόμαυρη εικόνα -> SLIC -> εξαγωγή ΙΔΙΩΝ features ->
# SVM -> φέρνουμε τις πιθανότητες στο πλήρες μέγεθος -> Graph Cut -> ξαναφτιάχνουμε Lab -> RGB.
#

import os
import numpy as np
import joblib
from skimage import io, color
from skimage.segmentation import slic
from skimage import graph
import gco 
from gabor_surf import extract_features_for_image 

print("--- Script Χρωματισμού (Πρόβλεψη) ---")

TEST_IMAGE_PATH = "data/blacknwhite.jpg"   
N_SUPERPIXELS = 400
COMPACTNESS = 10
LAMBDA_SMOOTH = 20.0     

# 1. Φόρτωση test εικόνας
img = io.imread(TEST_IMAGE_PATH)

# μπορεί να είναι (H,W) ή (H,W,3)
if img.ndim == 2:  # καθαρό γκρι
    gray = img.astype(np.float64) / 255.0
    # για να φτιάξουμε Lab χρειαζόμαστε 3 κανάλια -> τα κάνουμε stack
    rgb_for_vis = np.dstack([gray, gray, gray])
else:
    rgb_for_vis = img.astype(np.float64) / 255.0
    # αν ο εξεταστής δώσει έγχρωμη, την κάνουμε γκρι
    gray = color.rgb2gray(rgb_for_vis)

# παίρνουμε Lab ΜΟΝΟ για το L
lab_img = color.rgb2lab(np.dstack([gray, gray, gray]))
L_test = lab_img[:, :, 0]

h, w = L_test.shape
print(f"Test εικόνα: {h}x{w}")

# 2. SLIC στη test
segments_test = slic(
    np.dstack([gray, gray, gray]),
    n_segments=N_SUPERPIXELS,
    compactness=COMPACTNESS,
    channel_axis=-1,
    start_label=1,
)
valid_labels_test = np.unique(segments_test)
valid_labels_test = valid_labels_test[valid_labels_test > 0]
num_nodes = len(valid_labels_test)
print(f"Superpixels test: {num_nodes}")

# 3. Φόρτωση μοντέλων
kmeans = joblib.load("artifacts/kmeans_palette.joblib")
color_palette = kmeans.cluster_centers_
n_palette = kmeans.n_clusters
svm = joblib.load("artifacts/svm_colorizer.joblib")

# 4. Εξαγωγή χαρακτηριστικών ΜΕ ΑΚΡΙΒΩΣ ΤΟΝ ΙΔΙΟ ΤΡΟΠΟ
X_test, valid_labels_test = extract_features_for_image(L_test, segments_test)
print(f"X_test shape = {X_test.shape}")

# 5. Πρόβλεψη πιθανοτήτων
probs = svm.predict_proba(X_test)
classes_svm = svm.classes_
print(f"Το SVM ξέρει τις κλάσεις: {classes_svm}")

# ΠΡΟΒΛΗΜΑ: το SVM μπορεί να μην έχει δει όλες τις 0..n_palette-1
# Άρα πρέπει να "γεμίσουμε" τις υπόλοιπες με πολύ μικρή πιθανότητα
full_probabilities = np.full((num_nodes, n_palette), 1e-6, dtype=np.float32)
for idx, cls in enumerate(classes_svm):
    full_probabilities[:, cls] = probs[:, idx].astype(np.float32)

# κανονικοποίηση στο 1
full_probabilities /= full_probabilities.sum(axis=1, keepdims=True)

print(f"Προβλέψεις (με γέμισμα): {full_probabilities.shape}")

# 6. Graph Construction (RAG)
rag = graph.rag_mean_color(np.dstack([gray, gray, gray]), segments_test, mode="distance")
edges_from = []
edges_to = []
edge_weights = []

for n1, n2, data in rag.edges(data=True):
    # οι κόμβοι στο RAG ξεκινάνε από 1,1 κλπ, τους φέρνουμε σε 0..num_nodes-1
    if n1 == 0 or n2 == 0:
        # το 0 το αγνοούμε
        continue
    u = np.where(valid_labels_test == n1)[0][0]
    v = np.where(valid_labels_test == n2)[0][0]
    weight = data.get("weight", 1.0)
    # ομοιότητα -> κόστη
    edge_w = np.exp(-weight)   # όσο πιο κοντά χρώμα/ένταση, τόσο μεγαλύτερο βάρος
    edges_from.append(u)
    edges_to.append(v)
    edge_weights.append(edge_w)

edges = np.stack([np.array(edges_from, dtype=np.int32),
                  np.array(edges_to, dtype=np.int32)], axis=1)
edge_weights = np.array(edge_weights, dtype=np.float32)

print(f"[predict] Edges for graph cut: {edges.shape}")

# 7. Unary & Pairwise
# unary: (num_nodes, num_labels)
unary_cost = -np.log(full_probabilities + 1e-10).astype(np.float32)

# pairwise: Potts
pairwise_cost = (1.0 - np.eye(n_palette, dtype=np.float32)) * LAMBDA_SMOOTH

print("Εκτέλεση Graph Cut...")
labels = gco.cut_general_graph(unary_cost, pairwise_cost, edges, edge_weights)
print("Graph cut ολοκληρώθηκε.")

# 8. Ανακατασκευή a,b καναλιών
final_a = np.zeros((h, w), dtype=np.float64)
final_b = np.zeros((h, w), dtype=np.float64)

for sp_label, color_label in zip(valid_labels_test, labels):
    mask = (segments_test == sp_label)
    ab = color_palette[color_label]
    final_a[mask] = ab[0]
    final_b[mask] = ab[1]

final_lab = np.stack([L_test, final_a, final_b], axis=-1)
final_rgb = color.lab2rgb(final_lab, clip=True)

out_path = "colored_result.png"
io.imsave(out_path, (final_rgb * 255).astype(np.uint8))
print(f"Αποθηκεύτηκε στο {out_path}")

io.imshow(final_rgb)
io.show()
