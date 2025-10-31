import numpy as np
import joblib
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

print("Εκκίνηση...")

data = np.load("artifacts/training_features.npz")
X_train = data["X"]
y_train = data["y"]

print(f"X_train shape = {X_train.shape}")
print(f"y_train shape = {y_train.shape}")

kmeans = joblib.load("artifacts/kmeans_palette.joblib")
n_color_classes = kmeans.n_clusters
print(f"Παλέτα με {n_color_classes} κλάσεις.")

clf = make_pipeline(
    StandardScaler(),
    SVC(
        C=10.0,
        kernel="rbf",
        probability=True,  
        class_weight="balanced",
        gamma="scale",
    ),
)

clf.fit(X_train, y_train)
print("Το SVM εκπαιδεύτηκε.")

joblib.dump(clf, "artifacts/svm_colorizer.joblib")

probs = clf.predict_proba(X_train[:5])
print("Probabilities sample shape:", probs.shape)

for i in range(5):
    print(f"Sample {i}:")
    for cls, p in zip(clf.named_steps["svc"].classes_, probs[i]):
        print(f"  Class {cls}: {p:.4f}")
