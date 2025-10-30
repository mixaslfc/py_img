import numpy as np
import joblib
from skimage import io, color
from skimage.segmentation import slic
from skimage.filters import gabor
from scipy import ndimage as ndi
import gco 
from skimage import graph 

print("--- Script Χρωματισμού (Πρόβλεψη) ---")

# --- 0. Ρυθμίσεις ---
TEST_IMAGE_PATH = 'data/blacknwhite.jpg' # Βεβαιωθείτε ότι αυτή η διαδρομή είναι σωστή
N_SUPERPIXELS = 400 
COMPACTNESS = 10    
# --- ΥΠΕΡ-ΠΑΡΑΜΕΤΡΟΙ ΓΙΑ ΤΟ GRAPH CUT ---
LAMBDA = 10  # Βάρος Ομαλότητας
SIGMA = 10   # Ευαισθησία Ακμών
# --- 1. Φόρτωση Μοντέλων & Παλέτας ---
try:
    clf = joblib.load('svm_colorizer.joblib')
    kmeans = joblib.load('kmeans_palette.joblib')
except FileNotFoundError:
    print("Σφάλμα: Δεν βρέθηκαν τα αρχεία .joblib. Εκτελέσατε ξανά το svm_train.py;")
    exit()
color_palette = kmeans.cluster_centers_
K_classes = color_palette.shape[0] 
print(f"Τα μοντέλα SVM και K-Means φορτώθηκαν. Βρέθηκαν {K_classes} κλάσεις χρώματος.")
# --- 2. Φόρτωση & Επεξεργασία Εικόνας Δοκιμής (Βήμα i) ---
try:
    L_channel_test = io.imread(TEST_IMAGE_PATH, as_gray=True)
except FileNotFoundError:
    print(f"Σφάλμα: Δεν βρέθηκε η εικόνα δοκιμής στη διαδρομή: {TEST_IMAGE_PATH}")
    exit()
    
L_channel_test = L_channel_test * 100.0
h, w = L_channel_test.shape
print(f"Η εικόνα δοκιμής φορτώθηκε (Διαστάσεις: {h}x{w})")
# --- 3. SLIC Superpixels (Βήμα iii) ---
L_stacked = np.stack([L_channel_test, L_channel_test, L_channel_test], axis=-1)
segments_test = slic(L_stacked, 
                     n_segments=N_SUPERPIXELS, 
                     compactness=COMPACTNESS, 
                     channel_axis=-1,
                     start_label=1)
# --- 4. Εξαγωγή Χαρακτηριστικών (Βήμα iv) ---
print("Εξαγωγή χαρακτηριστικών από τη νέα εικόνα...")
valid_labels_test = np.unique(segments_test)
valid_labels_test = valid_labels_test[valid_labels_test > 0]
num_superpixels_test = len(valid_labels_test)
thetas = [0, np.pi/4, np.pi/2, 3*np.pi/4]
frequencies = [0.1, 0.5]
num_features_base = 2
num_gabor_features = len(thetas) * len(frequencies)
total_features = num_features_base + num_gabor_features
X_test_features = np.zeros((num_superpixels_test, total_features))
variance_L_test = ndi.variance(L_channel_test, labels=segments_test, index=valid_labels_test)
mean_L_test = ndi.mean(L_channel_test, labels=segments_test, index=valid_labels_test) 
X_test_features[:, 0] = mean_L_test
X_test_features[:, 1] = np.sqrt(np.maximum(0, variance_L_test))
feat_idx = num_features_base
for freq in frequencies:
    for theta in thetas:
        real, imag = gabor(L_channel_test, frequency=freq, theta=theta)
        response_mag = np.sqrt(real**2 + imag**2)
        X_test_features[:, feat_idx] = ndi.mean(response_mag, labels=segments_test, index=valid_labels_test)
        feat_idx += 1
print("Η εξαγωγή χαρακτηριστικών (X_test) ολοκληρώθηκε.")

# --- 5. Πρόβλεψη Πιθανοτήτων (Βήμα v) ---
# Ο πίνακας πιθανοτήτων που επιστρέφει το SVM (π.χ. σχήμα 368x30)
incomplete_probabilities = clf.predict_proba(X_test_features)
print(f"Οι (ελλιπείς) πιθανότητες προβλέφθηκαν. (Shape: {incomplete_probabilities.shape})")

# --- ΝΕΑ ΔΙΟΡΘΩΣΗ: "ΓΕΜΙΣΜΑ" ΤΩΝ ΠΙΘΑΝΟΤΗΤΩΝ ---
# Βρίσκουμε ποιες κλάσεις ΓΝΩΡΙΖΕΙ το SVM (π.χ. [0, 1, 3, ... 31])
# ΣΗΜΑΝΤΙΚΟ: Το 'clf' είναι pipeline, οπότε το SVC είναι το τελευταίο βήμα ('-1')
# και το πραγματικό αντικείμενο SVC είναι το δεύτερο στοιχείο ('1')
known_classes = clf.steps[-1][1].classes_
print(f"Το SVM γνωρίζει {len(known_classes)} από τις {K_classes} κλάσεις.")

# Δημιουργούμε έναν νέο, "πλήρη" πίνακα πιθανοτήτων (σχήμα 368x32)
# τον γεμίζουμε με μια πολύ μικρή πιθανότητα (epsilon)
full_probabilities = np.full((num_superpixels_test, K_classes), 1e-10)

# "Σκορπάμε" (scatter) τις πιθανότητες από τον ελλιπή πίνακα (368x30)
# στις σωστές στήλες του πλήρους πίνακα (368x32)
full_probabilities[:, known_classes] = incomplete_probabilities
# --- ΤΕΛΟΣ ΔΙΟΡΘΩΣΗΣ ---


# --- 6. Ανάθεση Χρώματος (Βήμα vi) - Graph Cuts ---
print("Προετοιμασία για Graph Cut...")

# 1. Unary Cost
# Χρησιμοποιούμε τον ΠΛΗΡΗ πίνακα πιθανοτήτων
unary_cost = -np.log(full_probabilities + 1e-10) # Χρησιμοποιούμε το 'full_probabilities'
unary_cost = unary_cost.astype(np.float32).copy(order='C')

# 2. Pairwise Cost
pairwise_cost_base = (1 - np.eye(K_classes, dtype=np.float32)) * LAMBDA
pairwise_cost = np.reshape(pairwise_cost_base, (K_classes, K_classes)).astype(np.float32).copy(order='C')

# 3. RAG & Edge Weights
label_to_index = {label: i for i, label in enumerate(valid_labels_test)}
rag = graph.rag_mean_color(L_channel_test, segments_test)
edges_from = []
edges_to = []
edge_weights = []
for u, v in rag.edges():
    if u in label_to_index and v in label_to_index:
        i = label_to_index[u]
        j = label_to_index[v]
        
        Li = mean_L_test[i]
        Lj = mean_L_test[j]
        
        L_diff_sq = (Li - Lj)**2
        weight = np.exp(-L_diff_sq / (2 * SIGMA**2))
        
        edges_from.append(i)
        edges_to.append(j)
        edge_weights.append(weight)
edges = np.array(edges_from, dtype=np.int32)
edges = np.stack((edges, np.array(edges_to, dtype=np.int32)), axis=1)
edge_weights = np.array(edge_weights, dtype=np.float32)

# 4. Επίλυση
print("Εκτέλεση α-expansion (Graph Cut)...")
# Debugging prints
print(f"Shape Unary Cost: {unary_cost.shape}, Type: {unary_cost.dtype}, Contiguous: {unary_cost.flags['C_CONTIGUOUS']}")
print(f"Shape Pairwise Cost: {pairwise_cost.shape}, Type: {pairwise_cost.dtype}, Contiguous: {pairwise_cost.flags['C_CONTIGUOUS']}")

# Έλεγχος αν οι διαστάσεις ταιριάζουν
if unary_cost.shape[1] != pairwise_cost.shape[0]:
    print(f"--- ΣΦΑΛΜΑ ΑΣΥΜΒΑΤΙΣΙΑΣ SHAPE ---")
    print(f"Unary columns ({unary_cost.shape[1]}) != Pairwise rows ({pairwise_cost.shape[0]})")
    exit()

final_labels = gco.cut_general_graph(unary_cost, pairwise_cost, edges, edge_weights)
print("Ο χρωματισμός ολοκληρώθηκε (μέθοδος Graph Cut).")

# --- 7. Ανακατασκευή & Αποθήκευση ---
final_a_channel = np.zeros((h, w), dtype=np.float64)
final_b_channel = np.zeros((h, w), dtype=np.float64)
for i, label_id in enumerate(valid_labels_test):
    color_label_index = final_labels[i] 
    color_ab = color_palette[color_label_index]
    mask = (segments_test == label_id)
    final_a_channel[mask] = color_ab[0]
    final_b_channel[mask] = color_ab[1]
final_lab_image = np.stack((L_channel_test, final_a_channel, final_b_channel), axis=-1)
final_rgb_image = color.lab2rgb(final_lab_image, ill='D65', clip=True)
output_filename = 'colored_result_graphcut.png'
io.imsave(output_filename, (final_rgb_image * 255).astype(np.uint8))
print(f"\n--- ✅ ΕΠΙΤΥΧΙΑ! ---")
print(f"Η χρωματισμένη εικόνα αποθηκεύτηκε στο: {output_filename}")
io.imshow(final_rgb_image)
io.show()