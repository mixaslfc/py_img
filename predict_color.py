import os
import numpy as np
import joblib
import matplotlib.pyplot as plt
from skimage import io, color
from skimage.segmentation import slic
from skimage import graph
import gco
from gabor_surf import extract_features_for_image

print("--- Script Χρωματισμού (Πρόβλεψη) ---")

TEST_IMAGE_PATH = "data/blacknwhite.jpg"
N_SUPERPIXELS = 200
COMPACTNESS = 10
LAMBDA_SMOOTH = 20               
EDGE_WEIGHT_SCALE = 200          

img = io.imread(TEST_IMAGE_PATH)

if img.ndim == 2:
    gray = img.astype(np.float64) / 255.0
    rgb_for_slic = np.dstack([gray, gray, gray])
else:
    rgb_for_slic = img.astype(np.float64) / 255.0
    gray = color.rgb2gray(rgb_for_slic)

lab_img = color.rgb2lab(np.dstack([gray, gray, gray]))
L_test = lab_img[:, :, 0]
h, w = L_test.shape
print(f"Test εικόνα: {h}x{w}")

segments_test = slic(
    rgb_for_slic,
    n_segments=N_SUPERPIXELS,
    compactness=COMPACTNESS,
    channel_axis=-1,
    start_label=1,
)
valid_labels_test = np.unique(segments_test)
valid_labels_test = valid_labels_test[valid_labels_test > 0]
num_nodes = len(valid_labels_test)
print(f"Superpixels test: {num_nodes}")

kmeans = joblib.load("artifacts/kmeans_palette.joblib")
color_palette = kmeans.cluster_centers_
n_labels = kmeans.n_clusters

svm = joblib.load("artifacts/svm_colorizer.joblib")

X_test, valid_labels_test = extract_features_for_image(L_test, segments_test)
print(f"X_test shape = {X_test.shape}")

probs = svm.predict_proba(X_test)
svm_classes = svm.classes_

full_probs = np.full((num_nodes, n_labels), 1e-6, dtype=np.float32)
for idx, cls in enumerate(svm_classes):
    full_probs[:, cls] = probs[:, idx].astype(np.float32)

full_probs /= full_probs.sum(axis=1, keepdims=True)
print(f"Προβλέψεις (με γέμισμα): {full_probs.shape}")

rag = graph.rag_mean_color(rgb_for_slic, segments_test, mode="distance")
edges_list = []
weights_list = []

for n1, n2, data in rag.edges(data=True):
    if n1 == 0 or n2 == 0:
        continue
    u = np.where(valid_labels_test == n1)[0][0]
    v = np.where(valid_labels_test == n2)[0][0]
    dist = data.get("weight", 1.0)
    sim = np.exp(-dist)
    edges_list.append([u, v])
    weights_list.append(sim)

edges = np.array(edges_list, dtype=np.int32)
edge_weights = np.array(weights_list, dtype=np.float32)

print(f"Edges for graph cut: {edges.shape}")

unary_cost = -np.log(full_probs + 1e-10)
unary_cost = (unary_cost * 100).astype(np.int32)
unary_cost = np.ascontiguousarray(unary_cost)

pairwise_cost = np.ones((n_labels, n_labels), dtype=np.int32) * LAMBDA_SMOOTH
np.fill_diagonal(pairwise_cost, 0)
pairwise_cost = np.ascontiguousarray(pairwise_cost)

edges = np.ascontiguousarray(edges, dtype=np.int32)
edge_weights = (edge_weights * EDGE_WEIGHT_SCALE).astype(np.int32)
edge_weights = np.ascontiguousarray(edge_weights)

# debug: αποθήκευση εικόνας μόνο με SVM για να δούμε διαφορά
print("Αποθήκευση εικόνας μόνο με SVM...")
svm_labels = full_probs.argmax(axis=1)
svm_a = np.zeros((h, w))
svm_b = np.zeros((h, w))
for sp_label, color_label in zip(valid_labels_test, svm_labels):
    mask = (segments_test == sp_label)
    ab = color_palette[color_label]
    svm_a[mask] = ab[0]
    svm_b[mask] = ab[1]
svm_lab = np.stack([L_test, svm_a, svm_b], axis=-1)
svm_rgb = color.lab2rgb(svm_lab)
io.imsave("colored_svm_only.png", (np.clip(svm_rgb,0,1)*255).astype(np.uint8))


print("Εκτέλεση Graph Cut...")
labels = gco.cut_general_graph(
    edges,
    edge_weights,
    unary_cost,
    pairwise_cost,
    n_iter=-1,
    algorithm="expansion",
)
print("Graph cut ολοκληρώθηκε.")

final_a = np.zeros((h, w), dtype=np.float64)
final_b = np.zeros((h, w), dtype=np.float64)

for sp_label, color_label in zip(valid_labels_test, labels):
    mask = (segments_test == sp_label)
    ab = color_palette[color_label]
    final_a[mask] = ab[0]
    final_b[mask] = ab[1]

final_lab = np.stack([L_test, final_a, final_b], axis=-1)
final_lab[:, :, 1] = np.clip(final_lab[:, :, 1], -110, 110)
final_lab[:, :, 2] = np.clip(final_lab[:, :, 2], -110, 110)
final_rgb = color.lab2rgb(final_lab)
final_rgb = np.clip(final_rgb, 0.0, 1.0)
out_path = "colored_result.png"
io.imsave(out_path, (final_rgb * 255).astype(np.uint8))


plt.imshow(final_rgb)
plt.show()