# svm_train.py
# ------------
# Βήμα v: εκπαιδεύουμε SVM πάνω στα features που έβγαλε το gabor_surf.py

import numpy as np
import joblib
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

print("Εκκίνηση...")

# 1. Φορτώνουμε τα X, y
data = np.load("artifacts/training_features.npz")
X_train = data["X"]
y_train = data["y"]

print(f"X_train shape = {X_train.shape}")
print(f"y_train shape = {y_train.shape}")

# 2. Φορτώνουμε και την παλέτα (για να ξέρουμε πόσες κλάσεις έχουμε)
kmeans = joblib.load("artifacts/kmeans_palette.joblib")
n_color_classes = kmeans.n_clusters
print(f"Παλέτα με {n_color_classes} κλάσεις.")

# 3. Φτιάχνουμε pipeline: StandardScaler -> SVC
clf = make_pipeline(
    StandardScaler(),
    SVC(
        C=10.0,
        kernel="rbf",
        probability=True,   # ΠΟΛΥ ΣΗΜΑΝΤΙΚΟ για το graph cut
        class_weight="balanced",
        gamma="scale",
    ),
)

clf.fit(X_train, y_train)
print("Το SVM εκπαιδεύτηκε.")

# 4. Αποθήκευση
joblib.dump(clf, "artifacts/svm_colorizer.joblib")
print("Αποθηκεύτηκε στο artifacts/svm_colorizer.joblib")

# 5. Μικρό check
probs = clf.predict_proba(X_train[:5])
print("Probabilities sample shape:", probs.shape)
