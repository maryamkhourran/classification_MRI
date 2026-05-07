"""
================================================================
  Classification MRI Cérébrale — K-Means FROM SCRATCH
  Pipeline complet avec segmentation avancée + visualisation
  nuages de points (brut → K-Means → vrais labels)
================================================================
Dataset :
    Training/
        glioma/ | meningioma/ | notumor/ | pituitary/
"""
# This script is the reference multiclass academic pipeline used by the web
# application for K-Means (k=4). It preprocesses the MRI images, segments
# the region of interest, extracts handcrafted features, then clusters the
# cases into the four clinical classes.


import os, glob, warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
from PIL import Image
warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────
#  CONFIG
# ──────────────────────────────────────────────────────────────
IMG_SIZE    = 128
N_CLUSTERS  = 4
N_SAMPLES   = 50          # images par classe
KMEANS_ITER = 150
SEED        = 42
CLASSES     = ["glioma_tumor", "meningioma_tumor", "no_tumor", "pituitary_tumor"]
COLORS      = ["#E74C3C", "#3498DB", "#2ECC71", "#F39C12"]
CMAP_CUSTOM = ListedColormap(COLORS)

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
        files = (glob.glob(os.path.join(path, "*.jpg")) +
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
    print(f"\n  Total : {len(images_raw)} images  ({IMG_SIZE}×{IMG_SIZE} px, niveaux de gris)")

    # ── Visualisation : grille 4×5 d'exemples bruts ──
    fig, axes = plt.subplots(4, 5, figsize=(14, 12))
    fig.patch.set_facecolor("#1A1A2E")
    fig.suptitle("ÉTAPE 1 — Échantillons MRI bruts", fontsize=16,
                 fontweight="bold", color="white", y=1.01)
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
    plt.savefig("etape1_images_brutes.png", dpi=130, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print("  → Sauvegardé : etape1_images_brutes.png")
    return images_raw, labels


# ════════════════════════════════════════════════════════════════
#  ÉTAPE 2 — PRÉ-TRAITEMENT
# ════════════════════════════════════════════════════════════════

def gaussian_kernel(size=5, sigma=1.4):
    k = size // 2
    x, y = np.mgrid[-k:k+1, -k:k+1]
    g = np.exp(-(x**2 + y**2) / (2*sigma**2))
    return g / g.sum()

def conv2d(img, kernel):
    """Convolution 2D manuelle avec padding 'reflect'."""
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
    equalized = equalize_hist(np.clip(blurred,0,255).astype(np.uint8))
    normed    = normalize(equalized)
    return normed

def preprocess_all(images_raw, labels):
    print("\n" + "═"*65)
    print("  ÉTAPE 2 — Pré-traitement (flou gaussien + égalisation + norm.)")
    print("═"*65)
    processed = []
    for i, img in enumerate(images_raw):
        processed.append(preprocess(img))
        if (i+1) % 30 == 0:
            print(f"  ... {i+1}/{len(images_raw)}")
    processed = np.array(processed)

    # ── Visualisation : brut → flou → égalisé → normalisé ──
    fig, axes = plt.subplots(4, 4, figsize=(15, 14))
    fig.patch.set_facecolor("#1A1A2E")
    fig.suptitle("ÉTAPE 2 — Pipeline de pré-traitement par classe",
                 fontsize=14, fontweight="bold", color="white")
    col_titles = ["Brute (gris)", "Flou Gaussien", "Égalisation hist.", "Normalisée [0-1]"]
    for c, t in enumerate(col_titles):
        axes[0][c].set_title(t, color="white", fontsize=10, fontweight="bold", pad=6)

    for r, ci in enumerate(range(4)):
        idx  = np.where(labels == ci)[0][0]
        raw  = images_raw[idx]
        blur = np.clip(conv2d(raw, GK), 0, 255).astype(np.uint8)
        eq   = equalize_hist(blur)
        norm = normalize(eq)
        for c, (step, cm) in enumerate(zip([raw, blur, eq, norm],
                                            ["gray","gray","gray","inferno"])):
            ax = axes[r][c]
            ax.imshow(step, cmap=cm)
            ax.axis("off")
            if c == 0:
                ax.set_ylabel(CLASSES[ci], color=COLORS[ci],
                              fontsize=10, fontweight="bold")
    plt.tight_layout()
    plt.savefig("etape2_pretraitement.png", dpi=130, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print("  → Sauvegardé : etape2_pretraitement.png")
    return processed


# ════════════════════════════════════════════════════════════════
#  ÉTAPE 3 — SEGMENTATION AVANCÉE (comme figure V.5)
# ════════════════════════════════════════════════════════════════

def otsu_threshold(img_norm):
    """Seuillage d'Otsu from scratch."""
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
        mb = sb / wb; mf = (S - sb) / wf
        var = wb * wf * (mb - mf)**2
        if var > best_var: best_var, best_t = var, t
    return best_t / 255.0

def connected_components(mask):
    """
    Labellisation des composantes connexes (BFS) from scratch.
    Retourne : label_map, nombre de composantes
    """
    h, w = mask.shape
    labels = np.zeros((h,w), dtype=int)
    current_label = 0
    for i in range(h):
        for j in range(w):
            if mask[i,j] and labels[i,j] == 0:
                current_label += 1
                # BFS
                queue = [(i,j)]
                labels[i,j] = current_label
                while queue:
                    ci, cj = queue.pop(0)
                    for di, dj in [(-1,0),(1,0),(0,-1),(0,1)]:
                        ni, nj = ci+di, cj+dj
                        if 0<=ni<h and 0<=nj<w and mask[ni,nj] and labels[ni,nj]==0:
                            labels[ni,nj] = current_label
                            queue.append((ni,nj))
    return labels, current_label

def largest_component(mask):
    """Garde uniquement la plus grande composante connexe."""
    label_map, n = connected_components(mask)
    if n == 0:
        return np.zeros_like(mask)
    sizes = [(label_map == k).sum() for k in range(1, n+1)]
    best = np.argmax(sizes) + 1
    return (label_map == best).astype(bool)

def morpho_erode(mask, iters=2):
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

def morpho_dilate(mask, iters=3):
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

def segment_tumor_full(img_norm):
    """
    Pipeline de segmentation en 5 étapes :
    (a) Image filtrée (normalisée)
    (b) Seuillage Otsu → masque binaire grossier
    (c) Érosion morphologique → supprime le bruit
    (d) Plus grande composante connexe → région tumorale isolée
    (e) Dilatation → masque final propre
    """
    # (a) image pré-traitée déjà fournie
    # (b) Otsu
    thresh   = otsu_threshold(img_norm)
    high_p   = np.percentile(img_norm, 82)
    binary   = img_norm > max(thresh, high_p * 0.88)
    # (c) érosion
    eroded   = morpho_erode(binary, iters=1)
    # (d) composante connexe principale
    conn     = largest_component(eroded)
    # (e) dilatation finale
    final    = morpho_dilate(conn, iters=2)
    return binary, eroded, conn, final

def overlay_tumor(img_norm, mask, color=(1,0,0)):
    """Superpose le masque en couleur sur l'image en niveaux de gris."""
    rgb = np.stack([img_norm]*3, axis=-1)
    rgb[mask, 0] = color[0]
    rgb[mask, 1] = color[1]
    rgb[mask, 2] = color[2]
    return np.clip(rgb, 0, 1)

def visualize_segmentation(images_proc, labels):
    print("\n" + "═"*65)
    print("  ÉTAPE 3 — Segmentation avancée (détection de la tumeur)")
    print("═"*65)

    # ── Figure 1 : pipeline complet pour 1 image avec tumeur ──
    # Choisit un glioma comme exemple principal
    idx_tumor  = np.where(labels == 0)[0][2]   # glioma
    idx_sain   = np.where(labels == 2)[0][2]   # notumor

    for idx, title_suffix in [(idx_tumor, "— Cas avec tumeur (Glioma)"),
                               (idx_sain,  "— Cas sain (No Tumor)")]:
        img  = images_proc[idx]
        binary, eroded, conn, final = segment_tumor_full(img)

        fig, axes = plt.subplots(1, 5, figsize=(20, 5))
        fig.patch.set_facecolor("#0D0D0D")
        fig.suptitle(f"ÉTAPE 3 — Pipeline de segmentation {title_suffix}",
                     fontsize=13, fontweight="bold", color="white", y=1.03)

        panels = [
            (img,                        "gray",  "(a) Image filtrée"),
            (binary.astype(float),       "hot",   "(b) Seuillage Otsu"),
            (eroded.astype(float),       "hot",   "(c) Érosion morpho."),
            (conn.astype(float),         "hot",   "(d) Composante connexe"),
            (overlay_tumor(img, final),  None,    "(e) Détection finale\n(tumeur en rouge)"),
        ]
        for ax, (data, cm, label) in zip(axes, panels):
            if cm is None:
                ax.imshow(data)
            else:
                ax.imshow(data, cmap=cm)
            ax.set_title(label, color="white", fontsize=9.5, fontweight="bold", pad=5)
            ax.axis("off")
            for spine in ax.spines.values():
                spine.set_edgecolor("#444")

        plt.tight_layout()
        suffix = "tumor" if "Glioma" in title_suffix else "sain"
        fname  = f"etape3_segmentation_{suffix}.png"
        plt.savefig(fname, dpi=130, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close()
        print(f"  → Sauvegardé : {fname}")

    # ── Figure 2 : segmentation sur 1 exemple de chaque classe ──
    fig, axes = plt.subplots(4, 5, figsize=(20, 17))
    fig.patch.set_facecolor("#0D0D0D")
    fig.suptitle("ÉTAPE 3 — Segmentation par classe MRI",
                 fontsize=14, fontweight="bold", color="white")
    col_titles = ["(a) Filtrée", "(b) Otsu", "(c) Érosion", "(d) Connexe", "(e) Détection finale"]
    for c, t in enumerate(col_titles):
        axes[0][c].set_title(t, color="white", fontsize=10, fontweight="bold", pad=6)

    for r, ci in enumerate(range(4)):
        idx = np.where(labels == ci)[0][3]
        img = images_proc[idx]
        binary, eroded, conn, final = segment_tumor_full(img)

        panels = [img, binary.astype(float), eroded.astype(float),
                  conn.astype(float), overlay_tumor(img, final)]
        cms    = ["gray","hot","hot","hot", None]

        for c, (data, cm) in enumerate(zip(panels, cms)):
            ax = axes[r][c]
            if cm is None:
                ax.imshow(data)
            else:
                ax.imshow(data, cmap=cm)
            ax.axis("off")
            if c == 0:
                has = ci != 2
                ax.set_ylabel(f"{CLASSES[ci]}\n{'🔴 Tumeur' if has else '🟢 Sain'}",
                              color=COLORS[ci], fontsize=9, fontweight="bold")
            if c == 4 and ci != 2:
                # bounding box autour de la tumeur détectée
                rows = np.where(final.any(axis=1))[0]
                cols = np.where(final.any(axis=0))[0]
                if len(rows) > 0 and len(cols) > 0:
                    ax.add_patch(mpatches.Rectangle(
                        (cols.min(), rows.min()),
                        cols.max()-cols.min(), rows.max()-rows.min(),
                        linewidth=2, edgecolor="yellow", facecolor="none"))

    plt.tight_layout()
    plt.savefig("etape3_segmentation_classes.png", dpi=130, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print("  → Sauvegardé : etape3_segmentation_classes.png")


# ════════════════════════════════════════════════════════════════
#  ÉTAPE 4 — EXTRACTION DES FEATURES (10 features manuelles)
# ════════════════════════════════════════════════════════════════

from skimage.feature import graycomatrix, graycoprops
from scipy.ndimage import binary_erosion

def extract_features(images, labels):
    """
    Extraction de caractéristiques radiomiques optimisée.
    Garantit la cohérence entre entraînement et prédiction.
    """
    all_features = []

    for img in images:
        # 1. SEGMENTATION UNIFIÉE
        _, _, _, mask = segment_tumor_full(img)
        mask_sum = np.sum(mask)

        # 2. SÉCURITÉ MASQUE VIDE (Correction n°2)
        if mask_sum == 0:
            mask = np.ones_like(img, dtype=bool)
            roi_pixels = img.flatten()
            mask_sum = img.size
        else:
            roi_pixels = img[mask]

        # 3. BOUNDING BOX & GLCM ROBUSTE (Correction n°4)
        coords = np.argwhere(mask)
        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0) + 1
        roi_crop = img[y0:y1, x0:x1]
        
        # Normalisation locale pour une GLCM stable [0-255]
        denom = (roi_crop.max() - roi_crop.min() + 1e-8)
        roi_norm = (roi_crop - roi_crop.min()) / denom
        roi_8bit = (roi_norm * 255).astype(np.uint8)

        # GLCM multi-angles (Invariance rotationnelle)
        angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
        glcm = graycomatrix(roi_8bit, distances=[1], angles=angles, 
                            levels=256, symmetric=True, normed=True)

        # 4. STATISTIQUES OPTIMISÉES (Correction n°3)
        mean_roi = np.mean(roi_pixels)
        std_roi  = np.std(roi_pixels) + 1e-5
        diff     = roi_pixels - mean_roi # Calculé une seule fois

        # 5. MORPHOLOGIE PRÉCISE (Correction n°5)
        eroded = binary_erosion(mask)
        # Périmètre basé sur les pixels de bordure réels
        perimeter = np.sum(mask & ~eroded)
        compactness = (perimeter**2) / (mask_sum + 1e-5)

        # 6. VECTEUR FINAL (10 caractéristiques normalisées)
        feat_vector = [
            mean_roi,
            std_roi,
            mask_sum / mask.size, # Ratio de surface (Correction n°5 du précédent tour)
            np.mean(graycoprops(glcm, 'contrast')),
            np.mean(graycoprops(glcm, 'homogeneity')),
            np.mean(graycoprops(glcm, 'energy')),
            np.mean(graycoprops(glcm, 'correlation')),
            compactness,
            np.mean(diff**3) / (std_roi**3), # Skewness
            np.mean(diff**4) / (std_roi**4)  # Kurtosis
        ]

        all_features.append(feat_vector)

    return np.array(all_features)

def zscore(X):
    mu  = X.mean(axis=0)
    sig = X.std(axis=0); sig[sig==0] = 1
    return (X - mu) / sig, mu, sig

def pca_2d(X):
    """PCA manuelle → 2 premières composantes."""
    Xc  = X - X.mean(axis=0)
    cov = Xc.T @ Xc / len(Xc)
    vals, vecs = np.linalg.eigh(cov)
    idx = np.argsort(vals)[::-1]
    vecs = vecs[:, idx]
    explained = vals[idx] / vals.sum() * 100
    return Xc @ vecs[:,:2], vecs[:,:2], explained[:2]

def pca_3d(X):
    """PCA manuelle → 3 premières composantes."""
    Xc  = X - X.mean(axis=0)
    cov = Xc.T @ Xc / len(Xc)
    vals, vecs = np.linalg.eigh(cov)
    idx = np.argsort(vals)[::-1]
    vecs = vecs[:, idx]
    explained = vals[idx] / vals.sum() * 100
    return Xc @ vecs[:,:3], explained[:3]


# ════════════════════════════════════════════════════════════════
#  ÉTAPE 5 — K-MEANS FROM SCRATCH
# ════════════════════════════════════════════════════════════════

class KMeans:
    def __init__(self, k=4, max_iter=KMEANS_ITER, tol=1e-5):
        self.k, self.max_iter, self.tol = k, max_iter, tol
        self.centroids = None
        self.labels_   = None
        self.inertia_  = None
        self.inertia_history = []

    def _init_pp(self, X):
        rng = np.random.RandomState(SEED)
        c   = [X[rng.randint(len(X))]]
        for _ in range(1, self.k):
            d2  = np.array([min(np.sum((x-ci)**2) for ci in c) for x in X])
            p   = d2 / d2.sum()
            cp  = p.cumsum()
            r   = rng.rand()
            c.append(X[np.searchsorted(cp, r)])
        return np.array(c)

    def _assign(self, X):
        d = np.array([[np.sum((x-c)**2) for c in self.centroids] for x in X])
        return np.argmin(d, axis=1)

    def fit(self, X):
        self.centroids = self._init_pp(X)
        for it in range(self.max_iter):
            old = self.centroids.copy()
            lbl = self._assign(X)
            for k in range(self.k):
                pts = X[lbl==k]
                if len(pts): self.centroids[k] = pts.mean(axis=0)
            inertia = sum(np.sum((X[lbl==k]-self.centroids[k])**2) for k in range(self.k))
            self.inertia_history.append(inertia)
            if np.sqrt(np.sum((self.centroids-old)**2)) < self.tol:
                print(f"  ✔  Convergence à l'itération {it+1}")
                break
        self.labels_  = self._assign(X)
        self.inertia_ = self.inertia_history[-1]
        return self


def elbow(X_norm):
    print("\n" + "═"*65)
    print("  ÉTAPE 5a — Méthode du coude (K optimal)")
    print("═"*65)
    inertias = []
    for k in range(2, 9):
        km = KMeans(k=k, max_iter=40)
        km.fit(X_norm)
        inertias.append(km.inertia_)
        print(f"  k={k}  inertie={km.inertia_:.2f}")

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor("#1A1A2E")
    ax.set_facecolor("#16213E")
    ax.plot(range(2,9), inertias, "o-", color="#00D4FF", lw=2.5,
            markersize=9, markerfacecolor="#E74C3C", markeredgecolor="white", mew=1.5)
    ax.axvline(x=4, color="#F39C12", ls="--", lw=1.8, label="k=4 (coude optimal)")
    ax.fill_between(range(2,9), inertias, alpha=0.15, color="#00D4FF")
    ax.set_xlabel("Nombre de clusters k", color="white", fontsize=12)
    ax.set_ylabel("Inertie (WCSS)", color="white", fontsize=12)
    ax.set_title("Méthode du coude — Choix du k optimal",
                 color="white", fontsize=13, fontweight="bold")
    ax.tick_params(colors="white"); ax.legend(fontsize=10)
    ax.grid(True, alpha=0.25, ls="--", color="white")
    ax.set_xticks(range(2,9))
    for spine in ax.spines.values(): spine.set_edgecolor("#444")
    plt.tight_layout()
    plt.savefig("etape5a_coude.png", dpi=130, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print("  → Sauvegardé : etape5a_coude.png")


# ════════════════════════════════════════════════════════════════
#  ÉTAPE 6 — NUAGES DE POINTS (3 phases)
# ════════════════════════════════════════════════════════════════

def align_clusters(km_labels, true_labels, k=4):
    from itertools import permutations
    best_acc, best_map = 0, None
    for perm in permutations(range(k)):
        mapped = np.array([perm[l] for l in km_labels])
        acc = (mapped == true_labels).mean()
        if acc > best_acc:
            best_acc, best_map = acc, perm
    return best_acc, best_map

def scatter_phase(ax, X2d, colors_per_point, title, centroids_2d=None,
                  legend_labels=None, bg="#16213E"):
    ax.set_facecolor(bg)
    ax.scatter(X2d[:,0], X2d[:,1],
               c=colors_per_point, s=28, alpha=0.75, edgecolors="none")
    if centroids_2d is not None:
        ax.scatter(centroids_2d[:,0], centroids_2d[:,1],
                   marker="*", s=350, c="white", edgecolors="#2C3E50",
                   linewidths=1.5, zorder=10, label="Centroids")
        ax.legend(fontsize=8, framealpha=0.6)
    if legend_labels:
        handles = [mpatches.Patch(color=COLORS[i], label=legend_labels[i])
                   for i in range(len(legend_labels))]
        ax.legend(handles=handles, fontsize=8, framealpha=0.6,
                  facecolor="#1A1A2E", labelcolor="white")
    ax.set_title(title, color="white", fontsize=11, fontweight="bold", pad=8)
    ax.set_xlabel("PC1", color="white", fontsize=9)
    ax.set_ylabel("PC2", color="white", fontsize=9)
    ax.tick_params(colors="white")
    ax.grid(True, alpha=0.2, ls="--", color="white")
    for spine in ax.spines.values(): spine.set_edgecolor("#444")

def visualize_scatter_3phases(km, X_norm, labels):
    """
    3 nuages de points côte à côte :
    Phase 1 — Données brutes (sans label, 1 seule couleur)
    Phase 2 — Après K-Means (couleur = cluster assigné)
    Phase 3 — Vrais labels (couleur = classe réelle)
    """
    print("\n" + "═"*65)
    print("  ÉTAPE 6 — Nuages de points : Brut → K-Means → Vrais labels")
    print("═"*65)

    X2d, vecs, expl = pca_2d(X_norm)

    # centroids projetés en 2D
    Xc_mean = X_norm.mean(axis=0)
    c_2d    = (km.centroids - Xc_mean) @ vecs

    # couleurs
    colors_raw   = ["#00D4FF"] * len(X2d)                     # Phase 1 : uniforme
    colors_km    = [COLORS[k] for k in km.labels_]            # Phase 2 : clusters
    colors_true  = [COLORS[l] for l in labels]                 # Phase 3 : classes réelles

    # alignement
    acc, best_map = align_clusters(km.labels_, labels)
    mapped = np.array([best_map[l] for l in km.labels_])
    correct = (mapped == labels)

    fig, axes = plt.subplots(1, 3, figsize=(21, 7))
    fig.patch.set_facecolor("#1A1A2E")
    fig.suptitle(
        f"ÉTAPE 6 — Classification K-Means  |  PCA (PC1={expl[0]:.1f}%  PC2={expl[1]:.1f}%)",
        fontsize=14, fontweight="bold", color="white", y=1.02)

    # ── Phase 1 : Données brutes (pas de label)
    scatter_phase(axes[0], X2d, colors_raw,
                  "① Nuage brut\n(sans classification)")

    # ── Phase 2 : Résultat K-Means
    scatter_phase(axes[1], X2d, colors_km,
                  f"② Résultat K-Means (k={N_CLUSTERS})\nAccuracy ≈ {acc*100:.1f}%",
                  centroids_2d=c_2d,
                  legend_labels=[f"Cluster {k}" for k in range(N_CLUSTERS)])

    # ── Phase 3 : Vrais labels
    scatter_phase(axes[2], X2d, colors_true,
                  "③ Vrais labels (classes réelles)\n(référence terrain)",
                  legend_labels=CLASSES)

    plt.tight_layout()
    plt.savefig("etape6_nuages_3phases.png", dpi=130, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"  → Sauvegardé : etape6_nuages_3phases.png  (accuracy={acc*100:.1f}%)")

    # ── Figure bonus : correct vs incorrect
    fig2, ax2 = plt.subplots(figsize=(9, 7))
    fig2.patch.set_facecolor("#1A1A2E")
    ax2.set_facecolor("#16213E")
    c_ok  = X2d[correct]
    c_err = X2d[~correct]
    ax2.scatter(c_ok[:,0],  c_ok[:,1],  c="#2ECC71", s=32, alpha=0.8,
                edgecolors="none", label=f"Correct ({correct.sum()})")
    ax2.scatter(c_err[:,0], c_err[:,1], c="#E74C3C", s=32, alpha=0.8,
                edgecolors="none", label=f"Incorrect ({(~correct).sum()})")
    ax2.scatter(c_2d[:,0], c_2d[:,1], marker="*", s=350, c="white",
                edgecolors="#2C3E50", linewidths=1.5, zorder=10, label="Centroids")
    ax2.set_title("Correct ✔ vs Incorrect ✘ (après alignement K-Means)",
                  color="white", fontsize=12, fontweight="bold")
    ax2.set_xlabel("PC1", color="white"); ax2.set_ylabel("PC2", color="white")
    ax2.tick_params(colors="white")
    ax2.legend(fontsize=10, facecolor="#1A1A2E", labelcolor="white")
    ax2.grid(True, alpha=0.2, ls="--", color="white")
    for spine in ax2.spines.values(): spine.set_edgecolor("#444")
    plt.tight_layout()
    plt.savefig("etape6b_correct_vs_incorrect.png", dpi=130, bbox_inches="tight",
                facecolor=fig2.get_facecolor())
    plt.close()
    print("  → Sauvegardé : etape6b_correct_vs_incorrect.png")
    return acc, best_map


# ════════════════════════════════════════════════════════════════
#  ÉTAPE 7 — ÉVALUATION COMPLÈTE
# ════════════════════════════════════════════════════════════════

def confusion_matrix(true, pred, k=4):
    cm = np.zeros((k,k), dtype=int)
    for t, p in zip(true, pred):
        cm[t,p] += 1
    return cm

def visualize_evaluation(km, X_norm, labels, features):
    print("\n" + "═"*65)
    print("  ÉTAPE 7 — Évaluation finale")
    print("═"*65)

    acc, best_map = align_clusters(km.labels_, labels)
    mapped = np.array([best_map[l] for l in km.labels_])
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

    # ── Figure : matrice de confusion + courbe convergence + radar ──
    fig, axes = plt.subplots(1, 3, figsize=(21, 7))
    fig.patch.set_facecolor("#1A1A2E")
    fig.suptitle("ÉTAPE 7 — Évaluation complète de la classification",
                 fontsize=14, fontweight="bold", color="white")

    # — Matrice de confusion —
    ax = axes[0]; ax.set_facecolor("#16213E")
    im = ax.imshow(cm, cmap="Blues")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(4)); ax.set_yticks(range(4))
    ax.set_xticklabels([c[:5] for c in CLASSES], rotation=30, ha="right",
                       color="white", fontsize=9)
    ax.set_yticklabels([c[:5] for c in CLASSES], color="white", fontsize=9)
    ax.set_xlabel("Prédit", color="white"); ax.set_ylabel("Réel", color="white")
    ax.set_title(f"Matrice de confusion\nAccuracy={acc*100:.1f}%",
                 color="white", fontweight="bold")
    cm_n = cm.astype(float) / (cm.sum(axis=1, keepdims=True)+1e-8)
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{cm[i,j]}", ha="center", va="center",
                    color="white" if cm_n[i,j]>0.4 else "#2C3E50",
                    fontweight="bold", fontsize=11)

    # — Courbe de convergence K-Means —
    ax2 = axes[1]; ax2.set_facecolor("#16213E")
    ax2.plot(km.inertia_history, color="#00D4FF", lw=2.5)
    ax2.fill_between(range(len(km.inertia_history)),
                     km.inertia_history, alpha=0.2, color="#00D4FF")
    ax2.set_xlabel("Itération", color="white", fontsize=11)
    ax2.set_ylabel("Inertie WCSS", color="white", fontsize=11)
    ax2.set_title("Convergence K-Means\n(inertie par itération)",
                  color="white", fontweight="bold")
    ax2.tick_params(colors="white")
    ax2.grid(True, alpha=0.25, ls="--", color="white")
    for spine in ax2.spines.values(): spine.set_edgecolor("#444")

    # — Radar features —
    ax3 = fig.add_subplot(1,3,3, polar=True)
    feat_names = ["Intens.","Std","Skew","Kurt","Entropie",
                  "S.Tumor","I.Tumor","Compact.","Gradient","Homog."]
    angles = np.linspace(0, 2*np.pi, 10, endpoint=False)
    angles = np.concatenate([angles, [angles[0]]])
    Xn, _, _ = zscore(features)
    Xn = (Xn - Xn.min()) / (Xn.max()-Xn.min()+1e-8)
    ax3.set_xticks(angles[:-1])
    ax3.set_xticklabels(feat_names, color="white", fontsize=8)
    ax3.tick_params(colors="white")
    ax3.set_facecolor("#16213E")
    for ci in range(4):
        vals = Xn[labels==ci].mean(axis=0).tolist() + [Xn[labels==ci].mean(axis=0)[0]]
        ax3.plot(angles, vals, color=COLORS[ci], lw=2, label=CLASSES[ci])
        ax3.fill(angles, vals, color=COLORS[ci], alpha=0.12)
    ax3.set_title("Profil moyen des features\npar classe", color="white",
                  fontweight="bold", pad=20)
    ax3.legend(loc="upper right", bbox_to_anchor=(1.4,1.15),
               fontsize=8, facecolor="#1A1A2E", labelcolor="white")

    plt.tight_layout()
    plt.savefig("etape7_evaluation.png", dpi=130, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print("  → Sauvegardé : etape7_evaluation.png")

    # ── Galerie prédictions ──
    fig2, axes2 = plt.subplots(4, 8, figsize=(22, 12))
    fig2.patch.set_facecolor("#0D0D0D")
    fig2.suptitle("ÉTAPE 7 — Galerie : Prédictions K-Means vs Vrais labels",
                  fontsize=13, fontweight="bold", color="white")
    idxs_all = []
    for ci in range(4):
        idxs_all.extend(np.where(labels==ci)[0][:8].tolist())
    idxs_all = idxs_all[:32]

    for ai, img_idx in enumerate(idxs_all):
        r2, c2 = ai//8, ai%8
        ax = axes2[r2][c2]
        ax.imshow(images_proc_global[img_idx], cmap="gray")
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
    fig2.legend(handles=handles, loc="lower center", ncol=2, fontsize=11,
                facecolor="#1A1A2E", labelcolor="white", bbox_to_anchor=(0.5,0.01))
    plt.tight_layout(); plt.subplots_adjust(bottom=0.07)
    plt.savefig("etape7_galerie.png", dpi=130, bbox_inches="tight",
                facecolor=fig2.get_facecolor())
    plt.close()
    print("  → Sauvegardé : etape7_galerie.png")


# ════════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════════

# variables globales utilisées dans les sous-fonctions
labels_global       = None
images_proc_global  = None

def main(dataset_path):
    global labels_global, images_proc_global

    print("\n╔" + "═"*63 + "╗")
    print("║   Classification MRI — K-Means FROM SCRATCH               ║")
    print("║   Segmentation avancée + Nuages de points (3 phases)      ║")
    print("╚" + "═"*63 + "╝")

    # 1. Chargement
    images_raw, labels = load_images(dataset_path)
    labels_global = labels

    # 2. Pré-traitement
    images_proc = preprocess_all(images_raw, labels)
    images_proc_global = images_proc

    # 3. Segmentation
    visualize_segmentation(images_proc, labels)

    # 4. Features
    features = extract_features(images_proc, labels)
    X_norm, mu, sig = zscore(features)

    # 5. Coude
    elbow(X_norm)

    # 5b. K-Means final
    print("\n" + "═"*65)
    print(f"  ÉTAPE 5b — K-Means (k={N_CLUSTERS}) from scratch")
    print("═"*65)
    km = KMeans(k=N_CLUSTERS, max_iter=KMEANS_ITER)
    km.fit(X_norm)

    # 6. Nuages de points 3 phases
    visualize_scatter_3phases(km, X_norm, labels)

    # 7. Évaluation
    visualize_evaluation(km, X_norm, labels, features)

    print("\n" + "═"*65)
    print("  ✅  Pipeline terminé ! Fichiers générés :")
    for f in ["etape1_images_brutes.png",
              "etape2_pretraitement.png",
              "etape3_segmentation_tumor.png",
              "etape3_segmentation_sain.png",
              "etape3_segmentation_classes.png",
              "etape5a_coude.png",
              "etape6_nuages_3phases.png",
              "etape6b_correct_vs_incorrect.png",
              "etape7_evaluation.png",
              "etape7_galerie.png"]:
        print(f"  → {f}")
    print("═"*65)


if __name__ == "__main__":
    import os
    # Change working directory to the folder where this script is located
    # so that "Training" folder and output images are always found/saved correctly
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    dataset_path = r"C:\Users\pc\Desktop\CNN\Training"
    main(dataset_path)
