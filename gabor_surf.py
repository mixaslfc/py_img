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

    if hasattr(cv2, "xfeatures2d") and hasattr(cv2.xfeatures2d, "SURF_create"):
        try:
            surf = cv2.xfeatures2d.SURF_create(hessianThreshold=400)
            # test call για να δούμε αν υλοποιήθηκε
            surf.detectAndCompute(np.zeros((20, 20), np.uint8), None)
            return "SURF", surf
        except cv2.error:
            pass
    # fallback σε ORB
    orb = cv2.ORB_create(nfeatures=500)
    return "ORB", orb


DESCRIPTOR_NAME, DESCRIPTOR_EXTRACTOR = _get_local_descriptor_extractor()
print(f"Θα χρησιμοποιηθεί τοπικός descriptor: {DESCRIPTOR_NAME}")


def _extract_descriptor_vector(gray_8u, mask, extractor_name, extractor_obj):
    keypoints, descriptors = extractor_obj.detectAndCompute(gray_8u, None)

    if keypoints is None or len(keypoints) == 0 or descriptors is None:
        if extractor_name == "SURF":
            return np.zeros(64, dtype=np.float32)
        else: 
            return np.zeros(32, dtype=np.float32)

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

    return selected_desc.mean(axis=0).astype(np.float32)


def extract_features_for_image(L_img, segments_img):
    unique_labels = np.unique(segments_img)
    valid_labels = unique_labels[unique_labels > 0]
    n_sp = len(valid_labels)

    n_gabor_feats = len(GABOR_THETAS) * len(GABOR_FREQUENCIES) * 2  
    desc_len = 64 if DESCRIPTOR_NAME == "SURF" else 32

    feature_dim = 2 + n_gabor_feats + desc_len + 2 
    X_img = np.zeros((n_sp, feature_dim), dtype=np.float32)

    gabor_real = []
    gabor_imag = []
    for th in GABOR_THETAS:
        for fr in GABOR_FREQUENCIES:
            real, imag = gabor(L_img, frequency=fr, theta=th)
            gabor_real.append(real)
            gabor_imag.append(imag)

    gray_8u = (L_img / L_img.max() * 255.0).astype(np.uint8)
    h, w = L_img.shape
    for i, sp_id in enumerate(valid_labels):
        mask = (segments_img == sp_id)
        ys, xs = np.where(mask)

        L_vals = L_img[mask]
        mean_L = L_vals.mean()
        std_L = L_vals.std()

        feats = [mean_L, std_L]

        for real, imag in zip(gabor_real, gabor_imag):
            feats.append(real[mask].mean())
            feats.append(imag[mask].mean())

        desc_vec = _extract_descriptor_vector(
            gray_8u,
            mask,
            DESCRIPTOR_NAME,
            DESCRIPTOR_EXTRACTOR,
        )
        feats.extend(desc_vec.tolist())

        mean_x = xs.mean() / w
        mean_y = ys.mean() / h
        feats.append(mean_x)
        feats.append(mean_y)

        X_img[i, :] = np.array(feats, dtype=np.float32)


    return X_img, valid_labels


slic_path = "artifacts/slic_segments.npz"
slic_npz = np.load(slic_path)
segments_list = [slic_npz[key] for key in sorted(slic_npz.files, key=lambda x: int(x.replace("arr_", "")))]

X_all = []
y_all = []

for idx, (lab_img, L_img, segments_img, qlabels_img) in enumerate(
    zip(training_labs, L_channels, segments_list, quantized_labels_list)
):

    X_img, valid_sp = extract_features_for_image(L_img, segments_img)

    y_img = np.zeros((X_img.shape[0],), dtype=np.int32)
    for i, sp_id in enumerate(valid_sp):
        mask = (segments_img == sp_id)
        label = mode(qlabels_img[mask], axis=None, keepdims=False)[0]
        y_img[i] = int(label)

    X_all.append(X_img)
    y_all.append(y_img)

X_all = np.vstack(X_all)
y_all = np.concatenate(y_all)

print(f"Feature matrix X shape: {X_all.shape}")
print(f"Label vector y shape: {y_all.shape}")

np.savez("artifacts/training_features.npz", X=X_all, y=y_all)

X_features = X_all
y_labels = y_all
