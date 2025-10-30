# gabor_surf.py
# -------------
# Βήμα iv: Εξαγωγή Χαρακτηριστικών Υφής (Gabor) ΚΑΙ τοπικών descriptors (SURF ή ORB) ανά superpixel
# από τις training εικόνες.
#
# Θα παραχθούν:
#   artifacts/training_features.npz   -> X (features), y (labels)
#
# Επίσης θα εξάγουμε τη συνάρτηση extract_features_for_image(...) για να την καλέσει το predict.

import os
import numpy as np
from scipy.stats import mode
from skimage.filters import gabor
from import_image import training_labs, L_channels
from lab import quantized_labels_list, color_palette
import cv2 


GABOR_THETAS = [0, np.pi/4, np.pi/2, 3*np.pi/4]
GABOR_FREQUENCIES = [0.2, 0.4]  


def _get_local_descriptor_extractor():
    """Επιστρέφει (name, extractor) όπου extractor έχει .detectAndCompute(...)"""
    # προσπαθούμε SURF
    if hasattr(cv2, "xfeatures2d") and hasattr(cv2.xfeatures2d, "SURF_create"):
        surf = cv2.xfeatures2d.SURF_create(hessianThreshold=400)
        return "SURF", surf
    # αλλιώς ORB (υπάρχει σχεδόν πάντα)
    orb = cv2.ORB_create(nfeatures=500)
    return "ORB", orb


DESCRIPTOR_NAME, DESCRIPTOR_EXTRACTOR = _get_local_descriptor_extractor()
print(f"Θα χρησιμοποιηθεί τοπικός descriptor: {DESCRIPTOR_NAME}")


def _extract_descriptor_vector(gray_8u, mask, extractor_name, extractor_obj):
    """
    Παίρνει μια gray εικόνα (uint8) + mask (bool) superpixel
    και επιστρέφει 1 διάνυσμα (π.χ. 64 για SURF, 32 για ORB).
    Αν δεν βρεθούν keypoints μέσα στο superpixel -> μηδενικό διάνυσμα.
    """
    keypoints, descriptors = extractor_obj.detectAndCompute(gray_8u, None)

    if keypoints is None or len(keypoints) == 0 or descriptors is None:
        # no keypoints
        if extractor_name == "SURF":
            return np.zeros(64, dtype=np.float32)
        else:  # ORB
            return np.zeros(32, dtype=np.float32)

    # κρατάμε μόνο όσα πέφτουν μέσα στο mask
    h, w = mask.shape
    selected_desc = []
    for kp, desc in zip(keypoints, descriptors):
        x, y = int(round(kp.pt[0])), int(round(kp.pt[1]))
        if 0 <= y < h and 0 <= x < w and mask[y, x]:
            selected_desc.append(desc)

    if not selected_desc:
        if extractor_name == "SURF":
            return np.zeros(64, dtype=np.float32)
        else:
            return np.zeros(32, dtype=np.float32)

    selected_desc = np.array(selected_desc, dtype=np.float32)

    # μέσος όρος descriptors → 1 διάνυσμα
    return selected_desc.mean(axis=0).astype(np.float32)


def extract_features_for_image(L_img, segments_img):
    """
    ΕΞΑΓΩΓΗ ΧΑΡΑΚΤΗΡΙΣΤΙΚΩΝ για εικόνα.
    Επιστρέφει:
        X_img: [N_superpixels, feature_dim]
        valid_labels: [N_superpixels]  (οι αριθμοί segment IDs)
    ΔΕΝ βάζει τα target y εδώ – αυτά τα παίρνουμε από το quantized_labels.
    """
    unique_labels = np.unique(segments_img)
    valid_labels = unique_labels[unique_labels > 0]
    n_sp = len(valid_labels)

    # υπολογίζουμε πόσα gabor features θα βγουν
    n_gabor_feats = len(GABOR_THETAS) * len(GABOR_FREQUENCIES) * 2  # real, imag
    # descriptor length
    desc_len = 64 if DESCRIPTOR_NAME == "SURF" else 32

    feature_dim = 2 + n_gabor_feats + desc_len  # meanL, stdL
    X_img = np.zeros((n_sp, feature_dim), dtype=np.float32)

    # προετοιμασία gabor bank (φίλτρα σε ΟΛΗ την εικόνα)
    gabor_real = []
    gabor_imag = []
    for th in GABOR_THETAS:
        for fr in GABOR_FREQUENCIES:
            real, imag = gabor(L_img, frequency=fr, theta=th)
            gabor_real.append(real)
            gabor_imag.append(imag)

    # gray για SURF/ORB
    gray_8u = (L_img / L_img.max() * 255.0).astype(np.uint8)

    for i, sp_id in enumerate(valid_labels):
        mask = (segments_img == sp_id)

        # 1) φωτεινότητα
        L_vals = L_img[mask]
        mean_L = L_vals.mean()
        std_L = L_vals.std()

        feats = [mean_L, std_L]

        # 2) gabor μέσα στο superpixel
        for real, imag in zip(gabor_real, gabor_imag):
            feats.append(real[mask].mean())
            feats.append(imag[mask].mean())

        # 3) SURF/ORB descriptor
        desc_vec = _extract_descriptor_vector(
            gray_8u,
            mask,
            DESCRIPTOR_NAME,
            DESCRIPTOR_EXTRACTOR,
        )
        feats.extend(desc_vec.tolist())

        X_img[i, :] = np.array(feats, dtype=np.float32)

    return X_img, valid_labels


# θα προσπαθήσουμε να διαβάσουμε τα SLIC από τον δίσκο (artifacts/slic_segments.npz)
# ώστε να μην τα ξαναυπολογίζουμε
slic_path = "artifacts/slic_segments.npz"
if not os.path.exists(slic_path):
    raise RuntimeError("❌ Δεν βρέθηκαν τα SLIC segments. Τρέξε πρώτα το slic.py")
slic_npz = np.load(slic_path)
segments_list = [slic_npz[key] for key in sorted(slic_npz.files, key=lambda x: int(x.replace("arr_", "")))]

X_all = []
y_all = []

for idx, (lab_img, L_img, segments_img, qlabels_img) in enumerate(
    zip(training_labs, L_channels, segments_list, quantized_labels_list)
):
    print(f"Επεξεργασία εικόνας εκπαίδευσης #{idx}...")

    X_img, valid_sp = extract_features_for_image(L_img, segments_img)
    # παίρνουμε το mode της κβάντισης μέσα στο superpixel
    y_img = np.zeros((X_img.shape[0],), dtype=np.int32)
    for i, sp_id in enumerate(valid_sp):
        mask = (segments_img == sp_id)
        label = mode(qlabels_img[mask], axis=None, keepdims=False)[0]
        y_img[i] = int(label)

    X_all.append(X_img)
    y_all.append(y_img)

X_all = np.vstack(X_all)
y_all = np.concatenate(y_all)

print(f"ΤΕΛΟΣ Βήματος iv.")
print(f"Feature matrix X shape: {X_all.shape}")
print(f"Label vector y shape: {y_all.shape}")

# αποθήκευση
np.savez("artifacts/training_features.npz", X=X_all, y=y_all)

X_features = X_all
y_labels = y_all
