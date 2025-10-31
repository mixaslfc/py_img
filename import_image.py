import os
import numpy as np
from skimage import io, color

TRAINING_IMAGE_PATHS = [
    "data/train_1.jpg",
    "data/train_2.jpg",
    "data/train_3.jpg",
    "data/train_4.jpg",
    "data/train_5.jpg",
    "data/train_6.jpg",
    "data/train_7.jpg",
    "data/train_8.jpg",
]

training_rgbs = []
training_labs = []
L_channels = []
a_channels = []
b_channels = []

for path in TRAINING_IMAGE_PATHS:

    rgb = io.imread(path)
    if rgb.dtype != np.float64:
        rgb = rgb.astype(np.float64) / 255.0
    lab = color.rgb2lab(rgb)

    training_rgbs.append(rgb)
    training_labs.append(lab)

    L = lab[:, :, 0]
    a = lab[:, :, 1]
    b = lab[:, :, 2]

    L_channels.append(L)
    a_channels.append(a)
    b_channels.append(b)

    print(f"Φορτώθηκε {path} με σχήμα {rgb.shape}")
    print(f"    Εύρος τιμών L: {L.min():.2f} έως {L.max():.2f}")
    print(f"    Εύρος τιμών a: {a.min():.2f} έως {a.max():.2f}")
    print(f"    Εύρος τιμών b: {b.min():.2f} έως {b.max():.2f}")

image_rgb = training_rgbs[0]
image_lab = training_labs[0]
L_channel = L_channels[0]
a_channel = a_channels[0]
b_channel = b_channels[0]

print(f"Συνολικές εικόνες εκπαίδευσης: {len(training_labs)}")

