"""
================================================================
  Classification MRI Cérébrale — CAH FROM SCRATCH
  Classification Ascendante Hiérarchique + Dendrogramme
  Pipeline complet : Chargement → Prétraitement → Segmentation
                     → Features → CAH → Dendrogramme
                     → Nuages de points (3 phases) → Évaluation
================================================================
Dataset :
    Training/
        glioma/ | meningioma/ | notumor/ | pituitary/
"""
# This script is the hierarchical-clustering counterpart of the multiclass
# K-Means pipeline. It now uses the same image-processing and feature-
# extraction logic so that the comparison between both methods stays fair.


import os, glob, warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from matplotlib.colors import ListedColormap
from PIL import Image
warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────
#  CONFIG
# ──────────────────────────────────────────────────────────────
IMG_SIZE   = 128
N_CLUSTERS = 4
N_SAMPLES  = 50        # images par classe (200 total)
SEED       = 42
CLASSES    = ["glioma", "meningioma", "notumor", "pituitary"]
COLORS     = ["#E74C3C", "#3498DB", "#2ECC71", "#F39C12"]

np.random.seed(SEED)


# ════════════════════════════════════════════════════════════════
#  ÉTAPE 1 — CHARGEMENT
# ════════════════════════════════════════════════════════════════

def load_images(dataset_path):
    print("\n" + "═"*65)
    print("  ÉTAPE 1 — Chargement des images MRI")
    print("═"*65)
    images_raw, labels = [], []

    for ci, cls in enumerate(CLASSES):
        path  = os.path.join(dataset_path, cls)
        files = (glob.glob(os.path.join(path, "*.jpg"))  +
                 glob.glob(os.path.join(path, "*.jpeg")) +
                 glob.glob(os.path.join(path, "*.png")))
        files = files[:N_SAMPLES]
        for f in files:
            img = Image.open(f).convert("L").resize((IMG_SIZE, IMG_SIZE))
            images_raw.append(np.array(img, dtype=np.uint8))
            labels.append(ci)
        print(f"  ✔  {cls:<12} → {len(files)} images")

    images_raw = np.array(images_raw)
    labels     = np.array(labels)
    print(f"\n  Total : {len(images_raw)} images  ({IMG_SIZE}×{IMG_SIZE} px)")

    # Visualisation grille d'exemples
    fig, axes = plt.subplots(4, 5, figsize=(14, 12))
    fig.patch.set_facecolor("#1A1A2E")
    fig.suptitle("ÉTAPE 1 — Échantillons MRI bruts",
                 fontsize=15, fontweight="bold", color="white")
    for r, ci in enumerate(range(4)):
        idxs = np.where(labels == ci)[0][:5]
        for c, idx in enumerate(idxs):
            ax = axes[r][c]
            ax.imshow(images_raw[idx], cmap="gray", vmin=0, vmax=255)
            ax.axis("off")
            if c == 0:
                ax.set_ylabel(CLASSES[ci], color=COLORS[ci],
                              fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig("cah_etape1_images_brutes.png", dpi=120,
                bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print("  → Sauvegardé : cah_etape1_images_brutes.png")
    return images_raw, labels


# ════════════════════════════════════════════════════════════════
#  ÉTAPE 2 — PRÉ-TRAITEMENT (from scratch)
# ════════════════════════════════════════════════════════════════

def gaussian_kernel(size=5, sigma=1.4):
    k = size // 2
    x, y = np.mgrid[-k:k+1, -k:k+1]
    g = np.exp(-(x**2 + y**2) / (2*sigma**2))
    return g / g.sum()

def conv2d(img, kernel):
    kh, kw = kernel.shape
    ph, pw = kh//2, kw//2
    padded = np.pad(img.astype(float), ((ph,ph),(pw,pw)), mode="reflect")
    out = np.zeros_like(img, dtype=float)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            out[i,j] = (padded[i:i+kh, j:j+kw] * kernel).sum()
    return out

def equalize_hist(img):
    hist, _ = np.histogram(img.flatten(), 256, (0,256))
    cdf = hist.cumsum()
    cdf_min = cdf[cdf>0].min()
    lut = np.round((cdf - cdf_min)/(img.size - cdf_min)*255).astype(np.uint8)
    return lut[img]

def normalize(img):
    mn, mx = img.min(), img.max()
    return (img.astype(float) - mn) / (mx - mn + 1e-8)

GK = gaussian_kernel(5, 1.4)

def preprocess(img_uint8):
    blurred   = conv2d(img_uint8, GK)
    equalized = equalize_hist(np.clip(blurred, 0, 255).astype(np.uint8))
    return normalize(equalized)

def preprocess_all(images_raw, labels):
    print("\n" + "═"*65)
    print("  ÉTAPE 2 — Pré-traitement (flou gaussien + égalisation + norm.)")
    print("═"*65)
    processed = []
    for i, img in enumerate(images_raw):
        processed.append(preprocess(img))
        if (i+1) % 40 == 0:
            print(f"  ... {i+1}/{len(images_raw)}")
    processed = np.array(processed)

    # Visualisation pipeline par classe
    fig, axes = plt.subplots(4, 4, figsize=(15, 14))
    fig.patch.set_facecolor("#1A1A2E")
    fig.suptitle("ÉTAPE 2 — Pipeline de pré-traitement par classe",
                 fontsize=13, fontweight="bold", color="white")
    for c, t in enumerate(["Brute", "Flou Gaussien", "Égalisation", "Normalisée"]):
        axes[0][c].set_title(t, color="white", fontsize=10, fontweight="bold")

    for r, ci in enumerate(range(4)):
        idx  = np.where(labels == ci)[0][0]
        raw  = images_raw[idx]
        blur = np.clip(conv2d(raw, GK), 0, 255).astype(np.uint8)
        eq   = equalize_hist(blur)
        norm = normalize(eq)
        for c, (step, cm) in enumerate(zip([raw, blur, eq, norm],
                                            ["gray","gray","gray","inferno"])):
            ax = axes[r][c]
            ax.imshow(step, cmap=cm); ax.axis("off")
            if c == 0:
                ax.set_ylabel(CLASSES[ci], color=COLORS[ci],
                              fontsize=10, fontweight="bold")
    plt.tight_layout()
    plt.savefig("cah_etape2_pretraitement.png", dpi=120,
                bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print("  → Sauvegardé : cah_etape2_pretraitement.png")
    return processed


# ════════════════════════════════════════════════════════════════
#  ÉTAPE 3 — SEGMENTATION AVANCÉE (from scratch)
# ════════════════════════════════════════════════════════════════

def otsu_threshold(img_norm):
    img8 = (img_norm * 255).astype(np.uint8)
    hist, _ = np.histogram(img8.flatten(), 256, (0,256))
    total = img8.size
    best_t, best_var = 0, -1
    wb, sb = 0, 0
    S = sum(i * hist[i] for i in range(256))
    for t in range(256):
        wb += hist[t]; wf = total - wb
        if wb == 0 or wf == 0: continue
        sb += t * hist[t]
        mb = sb/wb; mf = (S-sb)/wf
        var = wb * wf * (mb-mf)**2
        if var > best_var: best_var, best_t = var, t
    return best_t / 255.0

def largest_component_bfs(mask):
    """Composante connexe principale par BFS (from scratch)."""
    h, w = mask.shape
    labels_cc = np.zeros((h, w), dtype=int)
    current = 0
    for i in range(h):
        for j in range(w):
            if mask[i,j] and labels_cc[i,j] == 0:
                current += 1
                queue = [(i,j)]
                labels_cc[i,j] = current
                while queue:
                    ci, cj = queue.pop(0)
                    for di, dj in [(-1,0),(1,0),(0,-1),(0,1)]:
                        ni, nj = ci+di, cj+dj
                        if 0<=ni<h and 0<=nj<w and mask[ni,nj] and labels_cc[ni,nj]==0:
                            labels_cc[ni,nj] = current
                            queue.append((ni,nj))
    if current == 0:
        return np.zeros_like(mask)
    sizes = [(labels_cc==k).sum() for k in range(1, current+1)]
    return (labels_cc == np.argmax(sizes)+1).astype(bool)

def morpho_erode(mask, iters=1):
    result = mask.copy()
    for _ in range(iters):
        h, w = result.shape
        pad = np.pad(result, 1, mode="constant")
        new = np.ones((h,w), dtype=bool)
        for di in range(3):
            for dj in range(3):
                new &= pad[di:di+h, dj:dj+w]
        result = new
    return result

def morpho_dilate(mask, iters=2):
    result = mask.copy()
    for _ in range(iters):
        h, w = result.shape
        pad = np.pad(result, 1, mode="constant")
        new = np.zeros((h,w), dtype=bool)
        for di in range(3):
            for dj in range(3):
                new |= pad[di:di+h, dj:dj+w]
        result = new
    return result

def segment_tumor(img_norm):
    thresh  = otsu_threshold(img_norm)
    high_p  = np.percentile(img_norm, 82)
    binary  = img_norm > max(thresh, high_p * 0.88)
    eroded  = morpho_erode(binary, 1)
    conn    = largest_component_bfs(eroded)
    final   = morpho_dilate(conn, 2)
    return binary, eroded, conn, final

def overlay_tumor(img_norm, mask):
    rgb = np.stack([img_norm]*3, axis=-1)
    rgb[mask, 0] = 1.0
    rgb[mask, 1] = 0.0
    rgb[mask, 2] = 0.0
    return np.clip(rgb, 0, 1)

def visualize_segmentation(images_proc, labels):
    print("\n" + "═"*65)
    print("  ÉTAPE 3 — Segmentation avancée (détection tumeur)")
    print("═"*65)

    # Pipeline 5 étapes pour 1 exemple avec tumeur et 1 sain
    for ci_ex, suffix in [(0, "tumor"), (2, "sain")]:
        idx = np.where(labels == ci_ex)[0][2]
        img = images_proc[idx]
        binary, eroded, conn, final = segment_tumor(img)

        fig, axes = plt.subplots(1, 5, figsize=(22, 5))
        fig.patch.set_facecolor("#0D0D0D")
        title = "Cas avec tumeur (Glioma)" if suffix=="tumor" else "Cas sain (No Tumor)"
        fig.suptitle(f"ÉTAPE 3 — Segmentation : {title}",
                     fontsize=13, fontweight="bold", color="white")
        panels = [
            (img,                       "gray", "(a) Image filtrée"),
            (binary.astype(float),      "hot",  "(b) Seuillage Otsu"),
            (eroded.astype(float),      "hot",  "(c) Érosion morpho."),
            (conn.astype(float),        "hot",  "(d) Composante connexe"),
            (overlay_tumor(img, final), None,   "(e) Détection finale\n(tumeur en rouge)"),
        ]
        for ax, (data, cm, lbl) in zip(axes, panels):
            ax.imshow(data) if cm is None else ax.imshow(data, cmap=cm)
            ax.set_title(lbl, color="white", fontsize=9, fontweight="bold")
            ax.axis("off")
        plt.tight_layout()
        plt.savefig(f"cah_etape3_segmentation_{suffix}.png", dpi=120,
                    bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close()
        print(f"  → Sauvegardé : cah_etape3_segmentation_{suffix}.png")

    # Grille 4 classes
    fig, axes = plt.subplots(4, 5, figsize=(22, 18))
    fig.patch.set_facecolor("#0D0D0D")
    fig.suptitle("ÉTAPE 3 — Segmentation par classe MRI",
                 fontsize=13, fontweight="bold", color="white")
    for c, t in enumerate(["(a) Filtrée","(b) Otsu","(c) Érosion",
                            "(d) Connexe","(e) Détection finale"]):
        axes[0][c].set_title(t, color="white", fontsize=9, fontweight="bold")

    for r, ci in enumerate(range(4)):
        idx = np.where(labels == ci)[0][3]
        img = images_proc[idx]
        binary, eroded, conn, final = segment_tumor(img)
        panels = [img, binary.astype(float), eroded.astype(float),
                  conn.astype(float), overlay_tumor(img, final)]
        cms = ["gray","hot","hot","hot",None]
        for c, (data, cm) in enumerate(zip(panels, cms)):
            ax = axes[r][c]
            ax.imshow(data) if cm is None else ax.imshow(data, cmap=cm)
            ax.axis("off")
            if c == 0:
                has = ci != 2
                ax.set_ylabel(f"{CLASSES[ci]}\n{'🔴 Tumeur' if has else '🟢 Sain'}",
                              color=COLORS[ci], fontsize=9, fontweight="bold")
            if c == 4 and ci != 2:
                rows = np.where(final.any(axis=1))[0]
                cols = np.where(final.any(axis=0))[0]
                if len(rows) > 0 and len(cols) > 0:
                    ax.add_patch(mpatches.Rectangle(
                        (cols.min(), rows.min()),
                        cols.max()-cols.min(), rows.max()-rows.min(),
                        linewidth=2, edgecolor="yellow", facecolor="none"))
    plt.tight_layout()
    plt.savefig("cah_etape3_segmentation_classes.png", dpi=120,
                bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print("  → Sauvegardé : cah_etape3_segmentation_classes.png")

import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt

# --- ÉTAPE A : FONCTION UTILITAIRE (En dehors du processus principal) ---
def get_aligned_accuracy(y_true, y_pred):
    """Calcule l'accuracy après avoir aligné les clusters sur les vrais labels."""
    y_true = y_true.astype(np.int64)
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    row_ind, col_ind = linear_sum_assignment(w.max() - w)
    return sum([w[row_ind[i], col_ind[i]] for i in range(len(row_ind))]) / y_pred.size

# --- ÉTAPE B : LA FONCTION CAH (C'est ici qu'on met tout ton bloc) ---
def run_cah_analysis(X_raw, y_global):
    """
    Exécute le pipeline complet de la CAH.
    X_raw : les 10 features extraites
    y_global : les vrais labels pour l'évaluation
    """
    
    # 1. Prétraitement (Scaling + PCA)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)
    
    pca = PCA(n_components=0.90)
    X_pca = pca.fit_transform(X_scaled)
    
    # 2. Clustering
    Z = linkage(X_pca, method='ward', metric='euclidean')
    
    # 3. Découpage en 2 clusters
    clusters = fcluster(Z, t=2, criterion='maxclust')
    y_pred = clusters - 1 # Pour avoir 0 et 1 au lieu de 1 et 2
    
    # 4. Évaluation
    final_acc = get_aligned_accuracy(y_global, y_pred)
    final_ari = adjusted_rand_score(y_global, y_pred)
    final_sil = silhouette_score(X_pca, y_pred)
    
    # 5. Affichage des résultats
    print("\n" + "═"*40)
    print(f" RÉSULTATS CAH (Ward + PCA)")
    print("═"*40)
    print(f" Accuracy Alignée  : {final_acc:.2%}")
    print(f" Score ARI         : {final_ari:.3f}")
    print(f" Silhouette        : {final_sil:.3f}")
    print(f" Composantes PCA   : {pca.n_components_}")
    print("═"*40)
    
    # 6. Dendrogramme
    plt.figure(figsize=(10, 7))
    dendrogram(Z, truncate_mode='lastp', p=12)
    plt.title("Dendrogramme CAH (Méthode de Ward)")
    plt.show()
    
    return y_pred # On retourne les prédictions si besoin


# ════════════════════════════════════════════════════════════════
#  ÉTAPE 4 — EXTRACTION DES FEATURES (from scratch)
# ════════════════════════════════════════════════════════════════
from skimage.feature import graycomatrix, graycoprops
from scipy.ndimage import binary_erosion

def extract_features(images_proc, labels=None):
    """
    Extraction de caractéristiques radiomiques optimisée.
    Aligne CAH sur le même pipeline d'extraction que K-Means k=4.
    """
    features = []

    for img in images_proc:
        _, _, _, mask = segment_tumor(img)
        mask_sum = np.sum(mask)

        # Sécurité masque vide : on garde toute l'image comme ROI de secours.
        if mask_sum == 0:
            mask = np.ones_like(img, dtype=bool)
            roi_pixels = img.flatten()
            mask_sum = img.size
        else:
            roi_pixels = img[mask]

        coords = np.argwhere(mask)
        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0) + 1
        roi_crop = img[y0:y1, x0:x1]

        # Normalisation locale pour une GLCM stable [0-255].
        denom = (roi_crop.max() - roi_crop.min() + 1e-8)
        roi_norm = (roi_crop - roi_crop.min()) / denom
        roi_8bit = (roi_norm * 255).astype(np.uint8)

        angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
        glcm = graycomatrix(
            roi_8bit,
            distances=[1],
            angles=angles,
            levels=256,
            symmetric=True,
            normed=True
        )

        mean_roi = np.mean(roi_pixels)
        std_roi  = np.std(roi_pixels) + 1e-5
        diff = roi_pixels - mean_roi

        eroded = binary_erosion(mask)
        perimeter = np.sum(mask & ~eroded)
        compactness = (perimeter**2) / (mask_sum + 1e-5)

        features.append([
            mean_roi,
            std_roi,
            mask_sum / mask.size,
            np.mean(graycoprops(glcm, 'contrast')),
            np.mean(graycoprops(glcm, 'homogeneity')),
            np.mean(graycoprops(glcm, 'energy')),
            np.mean(graycoprops(glcm, 'correlation')),
            compactness,
            np.mean(diff**3) / (std_roi**3),
            np.mean(diff**4) / (std_roi**4)
        ])

    return np.array(features)

def zscore(X):
    mu  = X.mean(axis=0)
    sig = X.std(axis=0); sig[sig==0] = 1
    return (X - mu) / sig

def pca_2d(X):
    Xc  = X - X.mean(axis=0)
    cov = Xc.T @ Xc / len(Xc)
    vals, vecs = np.linalg.eigh(cov)
    idx = np.argsort(vals)[::-1]
    vecs = vecs[:, idx]
    expl = vals[idx] / vals.sum() * 100
    return Xc @ vecs[:,:2], expl[:2]


# ════════════════════════════════════════════════════════════════
#  ÉTAPE 5 — CAH FROM SCRATCH
# ════════════════════════════════════════════════════════════════

class CAH:
    """
    Classification Ascendante Hiérarchique — from scratch.

    Algorithme :
    1. Chaque point = 1 cluster
    2. Calculer la matrice de distances entre tous les clusters
    3. Fusionner les 2 clusters les plus proches
    4. Mettre à jour la matrice de distances (linkage)
    5. Répéter jusqu'à 1 seul cluster

    Linkages disponibles :
    - 'single'   : distance minimale (chaînage simple)
    - 'complete' : distance maximale (chaînage complet)
    - 'average'  : distance moyenne (UPGMA)
    - 'ward'     : minimise la variance intra-cluster (recommandé)
    """

    def __init__(self, n_clusters=4, linkage="ward"):
        self.n_clusters = n_clusters
        self.linkage    = linkage
        self.labels_    = None
        self.merge_history = []   # [(dist, cluster_A, cluster_B, nouveau_cluster)]

    def _dist_euclidean(self, a, b):
        return np.sqrt(np.sum((a - b)**2))

    def _update_distance_ward(self, dist_matrix, clusters, i, j, new_idx):
        """
        Mise à jour distance de Ward entre le nouveau cluster (i∪j) et chaque autre cluster k.
        Formule de Lance-Williams pour Ward :
        d(i∪j, k) = sqrt( (|k|+|i|)/(|k|+|i|+|j|) * d(k,i)²
                        + (|k|+|j|)/(|k|+|i|+|j|) * d(k,j)²
                        -  |k|/(|k|+|i|+|j|) * d(i,j)² )
        """
        ni = len(clusters[i])
        nj = len(clusters[j])
        new_dists = {}
        for k in dist_matrix:
            if k == i or k == j:
                continue
            nk  = len(clusters[k])
            dki = dist_matrix[k].get(i, dist_matrix[i].get(k, 0))
            dkj = dist_matrix[k].get(j, dist_matrix[j].get(k, 0))
            dij = dist_matrix[i].get(j, dist_matrix[j].get(i, 0))
            n_total = nk + ni + nj
            val = np.sqrt(max(0,
                ((nk+ni)/n_total) * dki**2 +
                ((nk+nj)/n_total) * dkj**2 -
                (nk/n_total)      * dij**2))
            new_dists[k] = val
        return new_dists

    def _update_distance_simple(self, dist_matrix, i, j, new_idx):
        new_dists = {}
        for k in dist_matrix:
            if k == i or k == j: continue
            dki = dist_matrix[k].get(i, dist_matrix[i].get(k, np.inf))
            dkj = dist_matrix[k].get(j, dist_matrix[j].get(k, np.inf))
            new_dists[k] = min(dki, dkj)
        return new_dists

    def _update_distance_complete(self, dist_matrix, i, j, new_idx):
        new_dists = {}
        for k in dist_matrix:
            if k == i or k == j: continue
            dki = dist_matrix[k].get(i, dist_matrix[i].get(k, 0))
            dkj = dist_matrix[k].get(j, dist_matrix[j].get(k, 0))
            new_dists[k] = max(dki, dkj)
        return new_dists

    def _update_distance_average(self, dist_matrix, clusters, i, j, new_idx):
        ni = len(clusters[i]); nj = len(clusters[j])
        new_dists = {}
        for k in dist_matrix:
            if k == i or k == j: continue
            dki = dist_matrix[k].get(i, dist_matrix[i].get(k, 0))
            dkj = dist_matrix[k].get(j, dist_matrix[j].get(k, 0))
            new_dists[k] = (ni*dki + nj*dkj) / (ni+nj)
        return new_dists

    def fit(self, X):
        n = len(X)
        print(f"\n  CAH ({self.linkage}) — {n} points, {n-1} fusions à effectuer...")

        # Initialisation : chaque point = son propre cluster
        clusters = {i: [i] for i in range(n)}
        # Centroids pour Ward
        centroids = {i: X[i].copy() for i in range(n)}

        # Matrice de distances initiale (symétrique → stockage triangulaire supérieur)
        print("  Calcul de la matrice de distances initiale...")
        dist_matrix = {}
        for i in range(n):
            dist_matrix[i] = {}
            for j in range(i+1, n):
                dist_matrix[i][j] = self._dist_euclidean(X[i], X[j])

        new_idx   = n    # index des nouveaux clusters fusionnés
        history   = []   # pour le dendrogramme

        step = 0
        while len(clusters) > self.n_clusters:
            # Trouver les 2 clusters les plus proches
            best_dist = np.inf
            best_i, best_j = -1, -1
            for i in dist_matrix:
                for j, d in dist_matrix[i].items():
                    if j in clusters and i in clusters and d < best_dist:
                        best_dist = d
                        best_i, best_j = i, j

            if best_i == -1:
                break

            # Fusionner best_i et best_j → new_idx
            merged = clusters[best_i] + clusters[best_j]
            history.append((best_dist, best_i, best_j, new_idx,
                            len(clusters[best_i]), len(clusters[best_j])))

            if step % 20 == 0:
                print(f"  ... fusion {step+1}/{n - self.n_clusters} "
                      f"| dist={best_dist:.4f} | clusters restants={len(clusters)-1}")
            step += 1

            # Mettre à jour distances
            if self.linkage == "ward":
                new_dists = self._update_distance_ward(
                    dist_matrix, clusters, best_i, best_j, new_idx)
            elif self.linkage == "single":
                new_dists = self._update_distance_simple(
                    dist_matrix, best_i, best_j, new_idx)
            elif self.linkage == "complete":
                new_dists = self._update_distance_complete(
                    dist_matrix, best_i, best_j, new_idx)
            else:  # average
                new_dists = self._update_distance_average(
                    dist_matrix, clusters, best_i, best_j, new_idx)

            # Supprimer les anciens clusters de la matrice
            del clusters[best_i]
            del clusters[best_j]
            if best_i in dist_matrix:
                del dist_matrix[best_i]
            if best_j in dist_matrix:
                del dist_matrix[best_j]
            for k in dist_matrix:
                dist_matrix[k].pop(best_i, None)
                dist_matrix[k].pop(best_j, None)

            # Ajouter le nouveau cluster
            clusters[new_idx] = merged
            if self.linkage == "ward":
                pts = X[merged]
                centroids[new_idx] = pts.mean(axis=0)
            dist_matrix[new_idx] = new_dists
            for k, d in new_dists.items():
                dist_matrix[k][new_idx] = d

            new_idx += 1

        # Assigner les labels finaux
        self.labels_ = np.zeros(n, dtype=int)
        for label, (cluster_id, members) in enumerate(clusters.items()):
            for m in members:
                self.labels_[m] = label

        self.merge_history = history
        print(f"  ✔  CAH terminée. {len(clusters)} clusters finaux.")
        return self


# ════════════════════════════════════════════════════════════════
#  ÉTAPE 6 — DENDROGRAMME (from scratch)
# ════════════════════════════════════════════════════════════════

def draw_dendrogram(cah, labels, n_show=60):
    """
    Dessine le dendrogramme from scratch à partir de merge_history.
    Affiche les n_show dernières fusions (les plus significatives).
    """
    print("\n" + "═"*65)
    print("  ÉTAPE 6 — Dendrogramme (visualisation hiérarchie)")
    print("═"*65)

    history = cah.merge_history
    # On ne montre que les dernières fusions (les plus hautes dans l'arbre)
    history_show = history[-n_show:]

    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    fig.patch.set_facecolor("#1A1A2E")

    # ── Dendrogramme simplifié (barres horizontales par niveau) ──
    ax = axes[0]
    ax.set_facecolor("#16213E")
    ax.set_title("Dendrogramme CAH — Dernières fusions",
                 color="white", fontsize=12, fontweight="bold")

    heights = [h[0] for h in history_show]
    x_pos   = range(len(heights))

    # Colorier les barres selon le niveau de fusion
    n = len(heights)
    cmap_dend = plt.cm.plasma
    for xi, (h, step_info) in enumerate(zip(heights, history_show)):
        color = cmap_dend(xi / max(n-1,1))
        ax.bar(xi, h, color=color, alpha=0.85, width=0.8, edgecolor="none")

    # Ligne de coupure à k=4 clusters (dernières 3 fusions = les plus hautes)
    if len(heights) >= 3:
        cut_height = (heights[-3] + heights[-4]) / 2 if len(heights) >= 4 else heights[-2]
        ax.axhline(y=cut_height, color="#F39C12", linewidth=2.5,
                   linestyle="--", label=f"Coupure → k={N_CLUSTERS} clusters")
        ax.legend(fontsize=9, facecolor="#1A1A2E", labelcolor="white")

    ax.set_xlabel("Étapes de fusion (ordre croissant de distance)",
                  color="white", fontsize=10)
    ax.set_ylabel("Distance de fusion (Ward)", color="white", fontsize=10)
    ax.tick_params(colors="white")
    for spine in ax.spines.values(): spine.set_edgecolor("#444")

    # ── Courbe des distances de fusion ──
    ax2 = axes[1]
    ax2.set_facecolor("#16213E")
    ax2.set_title("Courbe des distances — Choix du nombre de clusters",
                  color="white", fontsize=12, fontweight="bold")

    all_heights = [h[0] for h in cah.merge_history]
    ax2.plot(range(len(all_heights)), all_heights,
             color="#00D4FF", lw=2, alpha=0.8)
    ax2.fill_between(range(len(all_heights)), all_heights,
                     alpha=0.15, color="#00D4FF")

    # Marquer le saut le plus grand → endroit optimal de coupure
    diffs = np.diff(all_heights)
    if len(diffs) >= 4:
        top4 = np.argsort(diffs)[-4:]
        for idx in top4:
            ax2.axvline(x=idx+1, color="#E74C3C", lw=1.2, alpha=0.6, linestyle=":")

    # Marquer la coupure k=4
    cut_idx = len(all_heights) - N_CLUSTERS + 1
    if 0 <= cut_idx < len(all_heights):
        ax2.axvline(x=cut_idx, color="#F39C12", lw=2.5,
                    linestyle="--", label=f"Coupure k={N_CLUSTERS}")
        ax2.axhline(y=all_heights[cut_idx], color="#F39C12",
                    lw=1, linestyle=":", alpha=0.5)

    ax2.set_xlabel("Numéro de fusion", color="white", fontsize=10)
    ax2.set_ylabel("Distance de fusion", color="white", fontsize=10)
    ax2.tick_params(colors="white")
    ax2.legend(fontsize=9, facecolor="#1A1A2E", labelcolor="white")
    ax2.grid(True, alpha=0.2, ls="--", color="white")
    for spine in ax2.spines.values(): spine.set_edgecolor("#444")

    plt.tight_layout()
    plt.savefig("cah_etape6_dendrogramme.png", dpi=120,
                bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print("  → Sauvegardé : cah_etape6_dendrogramme.png")


# ════════════════════════════════════════════════════════════════
#  ÉTAPE 7 — NUAGES DE POINTS (3 phases)
# ════════════════════════════════════════════════════════════════

def align_clusters(cah_labels, true_labels, k=4):
    from itertools import permutations
    best_acc, best_map = 0, None
    for perm in permutations(range(k)):
        mapped = np.array([perm[l] for l in cah_labels])
        acc = (mapped == true_labels).mean()
        if acc > best_acc:
            best_acc, best_map = acc, perm
    return best_acc, best_map

def visualize_scatter_3phases(cah, X_norm, labels):
    print("\n" + "═"*65)
    print("  ÉTAPE 7 — Nuages de points : Brut → CAH → Vrais labels")
    print("═"*65)

    X2d, expl = pca_2d(X_norm)
    acc, best_map = align_clusters(cah.labels_, labels)
    mapped = np.array([best_map[l] for l in cah.labels_])

    colors_raw  = ["#00D4FF"] * len(X2d)
    colors_cah  = [COLORS[k] for k in cah.labels_]
    colors_true = [COLORS[l] for l in labels]

    fig, axes = plt.subplots(1, 3, figsize=(21, 7))
    fig.patch.set_facecolor("#1A1A2E")
    fig.suptitle(
        f"ÉTAPE 7 — Classification CAH  |  PCA (PC1={expl[0]:.1f}%  PC2={expl[1]:.1f}%)",
        fontsize=13, fontweight="bold", color="white", y=1.02)

    configs = [
        (colors_raw,  "① Nuage brut\n(sans classification)",     None),
        (colors_cah,  f"② Résultat CAH (k={N_CLUSTERS})\nAccuracy ≈ {acc*100:.1f}%",
                      [f"Cluster {k}" for k in range(N_CLUSTERS)]),
        (colors_true, "③ Vrais labels\n(référence terrain)",      CLASSES),
    ]

    for ax, (cols, title, leg_labels) in zip(axes, configs):
        ax.set_facecolor("#16213E")
        ax.scatter(X2d[:,0], X2d[:,1], c=cols, s=28, alpha=0.75, edgecolors="none")
        ax.set_title(title, color="white", fontsize=11, fontweight="bold", pad=8)
        ax.set_xlabel("PC1", color="white", fontsize=9)
        ax.set_ylabel("PC2", color="white", fontsize=9)
        ax.tick_params(colors="white")
        ax.grid(True, alpha=0.2, ls="--", color="white")
        for spine in ax.spines.values(): spine.set_edgecolor("#444")
        if leg_labels:
            handles = [mpatches.Patch(color=COLORS[i], label=leg_labels[i])
                       for i in range(len(leg_labels))]
            ax.legend(handles=handles, fontsize=8, framealpha=0.7,
                      facecolor="#1A1A2E", labelcolor="white")

    plt.tight_layout()
    plt.savefig("cah_etape7_nuages_3phases.png", dpi=120,
                bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  → Sauvegardé : cah_etape7_nuages_3phases.png  (accuracy={acc*100:.1f}%)")

    # Correct vs Incorrect
    correct = (mapped == labels)
    fig2, ax2 = plt.subplots(figsize=(9, 7))
    fig2.patch.set_facecolor("#1A1A2E")
    ax2.set_facecolor("#16213E")
    ax2.scatter(X2d[correct,0],  X2d[correct,1],  c="#2ECC71", s=30, alpha=0.8,
                edgecolors="none", label=f"Correct ({correct.sum()})")
    ax2.scatter(X2d[~correct,0], X2d[~correct,1], c="#E74C3C", s=30, alpha=0.8,
                edgecolors="none", label=f"Incorrect ({(~correct).sum()})")
    ax2.set_title("Correct ✔ vs Incorrect ✘ — CAH",
                  color="white", fontsize=12, fontweight="bold")
    ax2.set_xlabel("PC1", color="white"); ax2.set_ylabel("PC2", color="white")
    ax2.tick_params(colors="white")
    ax2.legend(fontsize=10, facecolor="#1A1A2E", labelcolor="white")
    ax2.grid(True, alpha=0.2, ls="--", color="white")
    for spine in ax2.spines.values(): spine.set_edgecolor("#444")
    plt.tight_layout()
    plt.savefig("cah_etape7b_correct_incorrect.png", dpi=120,
                bbox_inches="tight", facecolor=fig2.get_facecolor())
    plt.close()
    print("  → Sauvegardé : cah_etape7b_correct_incorrect.png")
    return acc, best_map


# ════════════════════════════════════════════════════════════════
#  ÉTAPE 8 — ÉVALUATION FINALE
# ════════════════════════════════════════════════════════════════

def confusion_matrix(true, pred, k=4):
    cm = np.zeros((k,k), dtype=int)
    for t, p in zip(true, pred):
        cm[t,p] += 1
    return cm

def visualize_evaluation(cah, X_norm, labels, features, images_proc):
    print("\n" + "═"*65)
    print("  ÉTAPE 8 — Évaluation finale")
    print("═"*65)

    acc, best_map = align_clusters(cah.labels_, labels)
    mapped = np.array([best_map[l] for l in cah.labels_])
    cm = confusion_matrix(labels, mapped)

    print(f"\n  Accuracy globale : {acc*100:.1f}%")
    print(f"  {'Classe':<14} {'Precision':>9} {'Recall':>9} {'F1':>9}")
    print(f"  {'─'*43}")
    for ci in range(N_CLUSTERS):
        tp = ((mapped==ci)&(labels==ci)).sum()
        fp = ((mapped==ci)&(labels!=ci)).sum()
        fn = ((mapped!=ci)&(labels==ci)).sum()
        pr = tp/(tp+fp+1e-8); rc = tp/(tp+fn+1e-8)
        f1 = 2*pr*rc/(pr+rc+1e-8)
        print(f"  {CLASSES[ci]:<14} {pr*100:8.1f}%  {rc*100:8.1f}%  {f1*100:8.1f}%")

    X2d, expl = pca_2d(X_norm)

    fig, axes = plt.subplots(1, 3, figsize=(21, 7))
    fig.patch.set_facecolor("#1A1A2E")
    fig.suptitle("ÉTAPE 8 — Évaluation complète de la CAH",
                 fontsize=13, fontweight="bold", color="white")

    # Matrice de confusion
    ax = axes[0]; ax.set_facecolor("#16213E")
    im = ax.imshow(cm, cmap="Blues")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(4)); ax.set_yticks(range(4))
    ax.set_xticklabels([c[:5] for c in CLASSES], rotation=30, ha="right",
                       color="white", fontsize=9)
    ax.set_yticklabels([c[:5] for c in CLASSES], color="white", fontsize=9)
    ax.set_xlabel("Prédit", color="white"); ax.set_ylabel("Réel", color="white")
    ax.set_title(f"Matrice de confusion\nAccuracy = {acc*100:.1f}%",
                 color="white", fontweight="bold")
    cm_n = cm.astype(float) / (cm.sum(axis=1,keepdims=True)+1e-8)
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{cm[i,j]}", ha="center", va="center",
                    color="white" if cm_n[i,j]>0.4 else "#2C3E50",
                    fontweight="bold", fontsize=11)

    # PCA coloré par cluster CAH
    ax2 = axes[1]; ax2.set_facecolor("#16213E")
    for k in range(N_CLUSTERS):
        mask = cah.labels_ == k
        ax2.scatter(X2d[mask,0], X2d[mask,1], c=COLORS[k], s=25,
                    alpha=0.7, edgecolors="none", label=f"Cluster {k}")
    ax2.set_title("Clusters CAH (PCA 2D)", color="white", fontweight="bold")
    ax2.set_xlabel("PC1", color="white"); ax2.set_ylabel("PC2", color="white")
    ax2.tick_params(colors="white")
    ax2.legend(fontsize=8, facecolor="#1A1A2E", labelcolor="white")
    ax2.grid(True, alpha=0.2, ls="--", color="white")
    for spine in ax2.spines.values(): spine.set_edgecolor("#444")

    # Radar features
    ax3 = fig.add_subplot(1,3,3, polar=True)
    feat_names = ["Intens.","Std","Skew","Kurt","Entropie",
                  "S.Tumor","I.Tumor","Compact.","Gradient","Homog."]
    angles = np.linspace(0, 2*np.pi, 10, endpoint=False)
    angles = np.concatenate([angles, [angles[0]]])
    Xn = zscore(features)
    Xn = (Xn - Xn.min()) / (Xn.max()-Xn.min()+1e-8)
    ax3.set_xticks(angles[:-1])
    ax3.set_xticklabels(feat_names, color="white", fontsize=8)
    ax3.tick_params(colors="white"); ax3.set_facecolor("#16213E")
    for ci in range(4):
        vals = Xn[labels==ci].mean(axis=0).tolist()
        vals += [vals[0]]
        ax3.plot(angles, vals, color=COLORS[ci], lw=2, label=CLASSES[ci])
        ax3.fill(angles, vals, color=COLORS[ci], alpha=0.12)
    ax3.set_title("Profil features par classe",
                  color="white", fontweight="bold", pad=20)
    ax3.legend(loc="upper right", bbox_to_anchor=(1.4,1.15),
               fontsize=8, facecolor="#1A1A2E", labelcolor="white")

    plt.tight_layout()
    plt.savefig("cah_etape8_evaluation.png", dpi=120,
                bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print("  → Sauvegardé : cah_etape8_evaluation.png")

    # Galerie prédictions
    fig3, axes3 = plt.subplots(4, 8, figsize=(22, 12))
    fig3.patch.set_facecolor("#0D0D0D")
    fig3.suptitle("ÉTAPE 8 — Galerie : Prédictions CAH vs Vrais labels",
                  fontsize=12, fontweight="bold", color="white")
    idxs_all = []
    for ci in range(4):
        idxs_all.extend(np.where(labels==ci)[0][:8].tolist())
    for ai, img_idx in enumerate(idxs_all[:32]):
        r2, c2 = ai//8, ai%8
        ax = axes3[r2][c2]
        ax.imshow(images_proc[img_idx], cmap="gray")
        true_c = labels[img_idx]; pred_c = mapped[img_idx]
        ok = true_c == pred_c
        for spine in ax.spines.values():
            spine.set_edgecolor("#2ECC71" if ok else "#E74C3C")
            spine.set_linewidth(3)
        ax.set_title(f"V:{CLASSES[true_c][:4]}\nP:{CLASSES[pred_c][:4]}",
                     fontsize=6.5, color="#2ECC71" if ok else "#E74C3C",
                     fontweight="bold")
        ax.axis("off")
    handles = [mpatches.Patch(color="#2ECC71", label="✔ Correct"),
               mpatches.Patch(color="#E74C3C", label="✘ Incorrect")]
    fig3.legend(handles=handles, loc="lower center", ncol=2,
                fontsize=11, facecolor="#1A1A2E", labelcolor="white",
                bbox_to_anchor=(0.5, 0.01))
    plt.tight_layout(); plt.subplots_adjust(bottom=0.07)
    plt.savefig("cah_etape8_galerie.png", dpi=120,
                bbox_inches="tight", facecolor=fig3.get_facecolor())
    plt.close()
    print("  → Sauvegardé : cah_etape8_galerie.png")


# ════════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════════

def main(dataset_path):
    print("\n╔" + "═"*63 + "╗")
    print("║  Classification MRI — CAH FROM SCRATCH                    ║")
    print("║  Dendrogramme + Nuages de points (3 phases)               ║")
    print("╚" + "═"*63 + "╝")

    # 1. Chargement
    images_raw, labels = load_images(dataset_path)
    if len(images_raw) == 0:
        print("\n❌ Aucune image trouvée ! Vérifiez le chemin du dataset.")
        return

    # 2. Pré-traitement
    images_proc = preprocess_all(images_raw, labels)

    # 3. Segmentation
    visualize_segmentation(images_proc, labels)

    # 4. Features
    features = extract_features(images_proc, labels)
    X_norm   = zscore(features)

    # 5. CAH
    print("\n" + "═"*65)
    print(f"  ÉTAPE 5 — CAH (Ward) from scratch — k={N_CLUSTERS}")
    print("═"*65)
    cah = CAH(n_clusters=N_CLUSTERS, linkage="ward")
    cah.fit(X_norm)

    # 6. Dendrogramme
    draw_dendrogram(cah, labels)
    
    draw_real_dendrogram(cah, len(X_norm))

    # 7. Nuages de points 3 phases
    visualize_scatter_3phases(cah, X_norm, labels)

    # 8. Évaluation
    visualize_evaluation(cah, X_norm, labels, features, images_proc)

    # Résumé correspondance clusters → classes
    print("\n" + "═"*65)
    print("  CORRESPONDANCE CLUSTERS CAH → CLASSES RÉELLES")
    print("═"*65)
    acc, best_map = align_clusters(cah.labels_, labels)
    print(f"\n  Cluster CAH    Classe assignée      Précision")
    print(f"  {'─'*52}")
    for cluster_idx in range(N_CLUSTERS):
        class_idx = best_map[cluster_idx]
        total   = (cah.labels_ == cluster_idx).sum()
        correct = ((cah.labels_ == cluster_idx) & (labels == class_idx)).sum()
        pct     = correct / total * 100 if total > 0 else 0
        print(f"  Cluster {cluster_idx}  →  {CLASSES[class_idx]:<16} "
              f"{pct:.1f}%  ({correct}/{total} images)")

    print(f"\n  Accuracy globale : {acc*100:.1f}%")
    print(f"\n  NOMS DES CLASSES POUR TON INTERFACE :")
    print(f"  {'─'*52}")
    descriptions = {
        "glioma":     "Tumeur gliale (maligne)",
        "meningioma": "Tumeur des méninges (souvent bénigne)",
        "notumor":    "Cerveau sain — aucune tumeur détectée",
        "pituitary":  "Tumeur de l'hypophyse",
        "glioma_tumor":     "Tumeur gliale (maligne)",
        "meningioma_tumor": "Tumeur des méninges (souvent bénigne)",
        "no_tumor":    "Cerveau sain — aucune tumeur détectée",
        "pituitary_tumor":  "Tumeur de l'hypophyse",
    }
    for i, cls in enumerate(CLASSES):
        desc = descriptions.get(cls, descriptions.get(cls.rstrip("_tumor"), "Description manquante"))
        print(f"  Classe {i}  →  {cls:<16} : {desc}")

    print("\n" + "═"*65)
    print("  ✅  Pipeline CAH terminé ! Fichiers générés :")
    for f in ["cah_etape1_images_brutes.png",
              "cah_etape2_pretraitement.png",
              "cah_etape3_segmentation_tumor.png",
              "cah_etape3_segmentation_sain.png",
              "cah_etape3_segmentation_classes.png",
              "cah_etape6_dendrogramme.png",
              "cah_etape7_nuages_3phases.png",
              "cah_etape7b_correct_incorrect.png",
              "cah_etape8_evaluation.png",
              "cah_etape8_galerie.png"]:
        print(f"  → {f}")
    print("═"*65)

# ════════════════════════════════════════════════════════════════
# VISUALISATION CLASSES CAH (PCA)
# ════════════════════════════════════════════════════════════════

def visualize_cah_clusters_pca(cah, X_norm, labels):
    print("\n" + "═"*65)
    print("  VISUALISATION — Classes CAH (PCA)")
    print("═"*65)

    X2d, expl = pca_2d(X_norm)

    fig, ax = plt.subplots(figsize=(9,7))
    fig.patch.set_facecolor("#1A1A2E")
    ax.set_facecolor("#16213E")

    for k in range(N_CLUSTERS):
        mask = cah.labels_ == k
        ax.scatter(X2d[mask,0], X2d[mask,1],
                   c=COLORS[k], s=30, alpha=0.8,
                   edgecolors="none", label=f"Cluster CAH {k}")

    ax.set_title("Visualisation PCA — Clusters CAH",
                 color="white", fontsize=12, fontweight="bold")
    ax.set_xlabel("PC1", color="white")
    ax.set_ylabel("PC2", color="white")
    ax.tick_params(colors="white")
    ax.legend(fontsize=9, facecolor="#1A1A2E", labelcolor="white")
    ax.grid(True, alpha=0.2, ls="--", color="white")

    for spine in ax.spines.values():
        spine.set_edgecolor("#444")

    plt.tight_layout()
    plt.savefig("cah_visualisation_clusters_pca.png", dpi=120,
                bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()

    print("  → Sauvegardé : cah_visualisation_clusters_pca.png")
    


# ════════════════════════════════════════════════════════════════
# GALERIE PAR CLUSTER CAH
# ════════════════════════════════════════════════════════════════

def gallery_by_cah_cluster(cah, images_proc, labels):
    print("\n" + "═"*65)
    print("  GALERIE — Images par cluster CAH")
    print("═"*65)

    fig, axes = plt.subplots(N_CLUSTERS, 8, figsize=(20, 10))
    fig.patch.set_facecolor("#0D0D0D")

    for k in range(N_CLUSTERS):
        idxs = np.where(cah.labels_ == k)[0][:8]

        for j in range(8):
            ax = axes[k][j]
            if j < len(idxs):
                idx = idxs[j]
                ax.imshow(images_proc[idx], cmap="gray")
                true_label = labels[idx]
                ax.set_title(f"V:{CLASSES[true_label][:4]}",
                             fontsize=7, color=COLORS[true_label])
            ax.axis("off")

        axes[k][0].set_ylabel(f"Cluster {k}",
                              color=COLORS[k],
                              fontsize=10,
                              fontweight="bold")

    plt.tight_layout()
    plt.savefig("cah_galerie_clusters.png", dpi=120,
                bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()

    print("  → Sauvegardé : cah_galerie_clusters.png")
        


def draw_real_dendrogram(cah, n_samples):
    print("\n" + "═"*65)
    print("  ÉTAPE 6 — VRAI Dendrogramme (style arbre)")
    print("═"*65)

    history = cah.merge_history
    n = n_samples

    # Construction de la matrice linkage (format scipy)
    Z = []
    cluster_sizes = {i:1 for i in range(n)}

    for dist, i, j, new_idx, size_i, size_j in history:
        Z.append([i, j, dist, size_i + size_j])
        cluster_sizes[new_idx] = size_i + size_j

    Z = np.array(Z)

    # Position des feuilles
    pos = {i: i for i in range(n)}
    heights = {i: 0 for i in range(n)}

    fig, ax = plt.subplots(figsize=(18, 7))
    fig.patch.set_facecolor("#1A1A2E")
    ax.set_facecolor("#16213E")

    for idx, (i, j, dist, size) in enumerate(Z):
        i, j = int(i), int(j)

        x1, x2 = pos[i], pos[j]
        y1, y2 = heights[i], heights[j]

        # nouvelle position = milieu
        x_new = (x1 + x2) / 2

        # dessiner les branches
        ax.plot([x1, x1], [y1, dist], color="white")
        ax.plot([x2, x2], [y2, dist], color="white")
        ax.plot([x1, x2], [dist, dist], color="white")

        # enregistrer nouveau cluster
        new_id = n + idx
        pos[new_id] = x_new
        heights[new_id] = dist

    ax.set_title("Dendrogramme CAH (Arbre hiérarchique)",
                 color="white", fontsize=14, fontweight="bold")

    ax.set_xlabel("Échantillons", color="white")
    ax.set_ylabel("Distance", color="white")

    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")

    plt.tight_layout()
    plt.savefig("cah_dendrogramme_reel.png", dpi=120,
                bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()

    print("  → Sauvegardé : cah_dendrogramme_reel.png")
    
    
    
    
    
if __name__ == "__main__":
    import os

    dataset_path = r"C:\Users\pc\Desktop\CNN\Training"

    # Affiche le contenu du dossier pour diagnostic
    print("\n  Contenu de ton dossier Training :")
    if os.path.isdir(dataset_path):
        for item in os.listdir(dataset_path):
            print(f"    - {item}")
    else:
        print(f"    Dossier introuvable : {dataset_path}")

    # Detection automatique des noms de sous-dossiers
    if os.path.isdir(dataset_path):
        found_dirs = sorted([
            d for d in os.listdir(dataset_path)
            if os.path.isdir(os.path.join(dataset_path, d))
        ])

        expected = ["glioma", "meningioma", "notumor", "pituitary"]
        matched  = []
        for exp in expected:
            found = None
            for d in found_dirs:
                if exp in d.lower() or d.lower() in exp:
                    found = d
                    break
            if exp == "notumor" and found is None:
                for d in found_dirs:
                    if ("no" in d.lower() and "tumor" in d.lower()) or                        "normal" in d.lower() or "healthy" in d.lower():
                        found = d
                        break
            matched.append(found)

        if all(matched):
            print("\n  Correspondance detectee :")
            for exp, real in zip(expected, matched):
                print(f"    {exp} -> {real}")
            CLASSES[:] = matched
            print(f"  Classes utilisees : {CLASSES}")
        else:
            print(f"\n  Dossiers trouves : {found_dirs}")
            print("  Assurez-vous que Training contient : glioma, meningioma, notumor, pituitary")

        output_dir = os.path.dirname(dataset_path)
        if output_dir:
            os.chdir(output_dir)
        main(dataset_path)