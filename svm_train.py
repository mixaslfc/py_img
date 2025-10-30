import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import joblib # Βιβλιοθήκη για αποθήκευση του εκπαιδευμένου μοντέλου

print("Βήμα v: Εκπαίδευση SVM...")

# 1. Εισαγωγή των X και y από το προηγούμενο βήμα
from gabor_surf import X_features as X_train_all
from gabor_surf import y_labels as y_train_all

# Έλεγχος ότι τα δεδομένα είναι καθαρά (όπως και πριν)
if np.isnan(X_train_all).any():
    print("ΣΦΑΛΜΑ: Τα δεδομένα εκπαίδευσης (X) περιέχουν NaN. Διακοπή.")
    exit()
if not np.isfinite(X_train_all).all():
    print("ΣΦΑΛΜΑ: Τα δεδομένα εκπαίδευσης (X) περιέχουν άπειρες τιμές. Διακοπή.")
    exit()
    
print(f"Φορτώθηκαν {X_train_all.shape[0]} δείγματα εκπαίδευσης με {X_train_all.shape[1]} χαρακτηριστικά.")

# 2. Δημιουργία του Pipeline
#    α) StandardScaler: Κανονικοποιεί τα features
#    β) SVC: Ο ταξινομητής SVM
#       - probability=True: ΑΠΑΡΑΙΤΗΤΟ για το Βήμα VI (Graph Cuts)
#       - kernel='rbf': (Radial Basis Function) Καλό για μη-γραμμικά δεδομένα
#       - C=1.0: Παράμετρος κανονικοποίησης (default, καλό για αρχή)
clf = make_pipeline(
    StandardScaler(),
    SVC(C=1.0, kernel='rbf', probability=True, random_state=42)
)

print("Το pipeline δημιουργήθηκε. Ξεκινά η εκπαίδευση (fit)...")
print("(Αυτό μπορεί να διαρκέσει μερικά λεπτά...)")

# 3. Εκπαίδευση του μοντέλου
clf.fit(X_train_all, y_train_all)

print("--- Η εκπαίδευση του SVM (Βήμα v) ολοκληρώθηκε! ---")

# 4. Αποθήκευση του εκπαιδευμένου μοντέλου
#    Αυτό είναι ΚΡΙΣΙΜΟ. Δεν θέλουμε να εκπαιδεύουμε το SVM
#    κάθε φορά που θέλουμε να χρωματίσουμε μια εικόνα.
model_filename = 'svm_colorizer.joblib'
joblib.dump(clf, model_filename)

print(f"Το εκπαιδευμένο μοντέλο αποθηκεύτηκε στο αρχείο: {model_filename}")

# 5. Έλεγχος πιθανοτήτων
#    Ας δούμε τι πιθανότητες δίνει για τα 5 πρώτα δείγματα
predicted_probabilities = clf.predict_proba(X_train_all[:5])
print("\nΔείγμα πιθανοτήτων για τα 5 πρώτα superpixels:")
print(f"(Shape: {predicted_probabilities.shape})")
# (Το shape θα είναι 5, N_COLOR_CLASSES)
print(predicted_probabilities)