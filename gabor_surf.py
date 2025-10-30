from skimage.filters import gabor
from scipy import ndimage as ndi
import numpy as np
from scipy.stats import mode 

# Εισαγωγή των μεταβλητών από τα προηγούμενα βήματα
from import_image import L_channel
from slic import segments
from lab import quantized_labels

# Παίρνουμε όλες τις μοναδικές ετικέτες...
unique_labels = np.unique(segments)
# ...και κρατάμε μόνο αυτές που είναι > 0 (οι έγκυρες ετικέτες superpixel)
valid_labels = unique_labels[unique_labels > 0]

num_superpixels = len(valid_labels)

num_features_base = 2 # Mean L, Std L
thetas = [0, np.pi/4, np.pi/2, 3*np.pi/4] # 4 γωνίες
frequencies = [0.1, 0.5] # 2 συχνότητες
num_gabor_features = len(thetas) * len(frequencies)
total_features = num_features_base + num_gabor_features

X_features = np.zeros((num_superpixels, total_features))
y_labels = np.zeros(num_superpixels, dtype=np.int32)

print("Βήμα iv: Ξεκινά η εξαγωγή χαρακτηριστικών (Στιβαρή έκδοση 2.0)...")

# 1. Βασικά χαρακτηριστικά (Mean L, Std L)
print("...Υπολογισμός βασικών χαρακτηριστικών (Mean L, Std L)...")
mean_L = ndi.mean(L_channel, labels=segments, index=valid_labels)

variance_L = ndi.variance(L_channel, labels=segments, index=valid_labels)
std_L = np.sqrt(np.maximum(0, variance_L)) # "Clamping" στο 0
# --- ΤΕΛΟΣ ΔΙΟΡΘΩΣΗΣ 2 ---

X_features[:, 0] = mean_L
X_features[:, 1] = std_L

# 2. Gabor Features
print("...Υπολογισμός Gabor features...")
feat_idx = num_features_base
for freq in frequencies:
    for theta in thetas:
        real, imag = gabor(L_channel, frequency=freq, theta=theta)
        response_mag = np.sqrt(real**2 + imag**2)
        
        mean_response = ndi.mean(response_mag, labels=segments, index=valid_labels)
        X_features[:, feat_idx] = mean_response
        feat_idx += 1

# 3. Δημιουργία ετικετών (Y)
print("...Δημιουργία ετικετών-στόχων (Y)...")
for i, label_id in enumerate(valid_labels):
    mask = (segments == label_id)
    
    label = mode(quantized_labels[mask], axis=None, keepdims=False)[0]
    
    y_labels[i] = label

print("--- Το Βήμα iv ολοκληρώθηκε ---")
print(f"Feature matrix X shape: {X_features.shape}")
print(f"Label vector y shape: {y_labels.shape}")
