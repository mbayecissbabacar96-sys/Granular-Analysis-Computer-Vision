# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 14:06:00 2026

@author: PC
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from skimage.draw import disk
from skimage.util import random_noise
from skimage.filters import gaussian, sobel, threshold_otsu
from skimage.measure import label, regionprops

# =========================
# 0. Initialisation
# =========================

output_dir = "output_figures"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

np.random.seed(42)  # Reproductibilité

# =========================
# 1. Génération image
# =========================

shape = (256, 256)
image = np.zeros(shape)

for _ in range(30):
    x, y = np.random.randint(20, 236, 2)
    r = np.random.randint(5, 15)
    rr, cc = disk((x, y), r, shape=shape)
    image[rr, cc] = 1

# =========================
# 2. Simulation expérimentale (bruit)
# =========================

noisy = random_noise(image, mode='gaussian', var=0.02)

# =========================
# 3. Filtrage
# =========================

filtered = gaussian(noisy, sigma=1.2)

# =========================
# 4. Détection de contours
# =========================

edges = sobel(filtered)

# =========================
# 5. Segmentation intelligente
# =========================

thresh = threshold_otsu(filtered)
binary = filtered > thresh
labels = label(binary)

# =========================
# 6. Analyse physique
# =========================

regions = regionprops(labels)
sizes = [r.area for r in regions]

# Statistiques (TRÈS IMPORTANT)
print("Nombre de grains détectés :", len(sizes))
print("Taille moyenne :", np.mean(sizes))
print("Écart-type :", np.std(sizes))

# =========================
# 7. Sauvegarde des figures
# =========================

figures_data = [
    (image, "1_modele_ideal", "gray"),
    (noisy, "2_image_bruitee", "gray"),
    (filtered, "3_filtrage_gaussien", "magma"),
    (edges, "4_contours_sobel", "gray"),
    (labels, "5_segmentation_grains", "nipy_spectral")
]

for data, name, cmap in figures_data:
    plt.figure(figsize=(6, 6))
    plt.imshow(data, cmap=cmap)
    plt.axis('off')
    plt.savefig(f"{output_dir}/{name}.png", bbox_inches='tight', dpi=300)
    plt.close()

# =========================
# 8. Histogramme amélioré
# =========================

plt.figure(figsize=(8, 6))
plt.hist(sizes, bins=12, density=True)  # normalisé
plt.title("Distribution des tailles de grains")
plt.xlabel("Aire (pixels)")
plt.ylabel("Densité de probabilité")
plt.savefig(f"{output_dir}/6_histogramme_tailles.png", dpi=300)
plt.close()

# =========================
# 9. Superposition contours (niveau publication)
# =========================

plt.figure(figsize=(6, 6))
plt.imshow(filtered, cmap='gray')
plt.contour(binary, colors='red')
plt.title("Contours détectés superposés")
plt.axis('off')
plt.savefig(f"{output_dir}/7_contours_superposes.png", dpi=300)
plt.close()

# =========================
# 10. Planche complète
# =========================

fig, ax = plt.subplots(2, 3, figsize=(15, 10))

ax[0, 0].imshow(image, cmap='gray')
ax[0, 0].set_title("1. Modèle idéal")

ax[0, 1].imshow(noisy, cmap='gray')
ax[0, 1].set_title("2. Image bruitée")

ax[0, 2].imshow(filtered, cmap='magma')
ax[0, 2].set_title("3. Filtrage gaussien")

ax[1, 0].imshow(edges, cmap='gray')
ax[1, 0].set_title("4. Contours (Sobel)")

ax[1, 1].imshow(labels, cmap='nipy_spectral')
ax[1, 1].set_title(f"5. Segmentation ({len(sizes)} grains)")

ax[1, 2].hist(sizes, bins=12, density=True)
ax[1, 2].set_title("6. Distribution")

for a in ax.flatten()[:5]:
    a.axis('off')

plt.tight_layout()
plt.savefig(f"{output_dir}/planche_complete_resultats.png", dpi=300)
plt.show()

print(f"✅ Terminé ! Les résultats sont dans le dossier : {output_dir}")
