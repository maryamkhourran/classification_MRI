"""Backend utilities for MRI preprocessing, caching, and page bundles.

This module is the bridge between the Flask routes and the academic scripts.
It knows where the datasets live, where generated figures are stored, how each
method is executed, and how the web pages receive their final payloads.
"""

import json
import os
import re
import subprocess
import sys
import time
from contextlib import redirect_stdout
from functools import lru_cache
from io import StringIO
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from werkzeug.datastructures import FileStorage
from werkzeug.utils import secure_filename


BASE_DIR = Path(__file__).resolve().parent
# Root folder that stores all MRI images used by the project.
PROJECT_DATA_DIR = BASE_DIR / "project_data"
# Folder containing the academic / algorithmic scripts.
PROJECT_SCRIPTS_DIR = BASE_DIR / "project_scripts"
# Shared multiclass dataset used by K-Means k=4 and CAH.
TRAIN_DIR = PROJECT_DATA_DIR / "multiclass" / "Training"
TEST_DIR = PROJECT_DATA_DIR / "multiclass" / "Testing"
# Binary tasks now reuse the exact same dataset; the label regrouping is done
# later in code, so we no longer need a duplicated "binary" dataset folder.
BINARY_TRAIN_DIR = TRAIN_DIR
BINARY_TEST_DIR = TEST_DIR
# Main script for K-Means multiclass classification.
KMEANS_SCRIPT = PROJECT_SCRIPTS_DIR / "mri_kmeans_classification_multiclass.py"
# Main script for K-Means binary classification.
KMEANS_BINARY_SCRIPT = PROJECT_SCRIPTS_DIR / "mri_kmeans_classification_binary.py"
# Main script for hierarchical clustering (CAH).
CAH_SCRIPT = PROJECT_SCRIPTS_DIR / "mri_cah_classification.py"

STATIC_DIR = BASE_DIR / "static"
GENERATED_DIR = STATIC_DIR / "generated"
UPLOAD_DIR = STATIC_DIR / "uploads"
KMEANS_OUTPUT_DIR = GENERATED_DIR / "kmeans"
KMEANS_BINARY_OUTPUT_DIR = GENERATED_DIR / "kmeans_binary"
CAH_OUTPUT_DIR = GENERATED_DIR / "cah"
DATA_DEMO_OUTPUT_DIR = GENERATED_DIR / "data_demo"
CACHE_DIR = BASE_DIR / ".cache"
CAH_SUMMARY_PATH = CACHE_DIR / "cah_summary.json"
DATA_DEMO_FEATURES_PATH = CACHE_DIR / "data_demo_feature_rows.json"
KMEANS_BINARY_RUNTIME_CACHE = CACHE_DIR / "kmeans_binary_runtime.npz"

# Canonical class names used by the whole project.
CLASSES = ["glioma_tumor", "meningioma_tumor", "no_tumor", "pituitary_tumor"]
KMEANS_IMAGE_ORDER = [
    "etape1_images_brutes.png",
    "etape2_pretraitement.png",
    "etape3_segmentation_tumor.png",
    "etape3_segmentation_sain.png",
    "etape3_segmentation_classes.png",
    "etape5a_coude.png",
    "etape6_nuages_3phases.png",
    "etape6b_correct_vs_incorrect.png",
    "etape7_evaluation.png",
    "etape7_galerie.png",
]
KMEANS_BINARY_IMAGE_ORDER = [
    "etape1_images_brutes.png",
    "etape2_pretraitement.png",
    "etape3_segmentation_tumor.png",
    "etape3_segmentation_sain.png",
    "etape3_segmentation_classes.png",
    "etape5a_coude.png",
    "etape6_nuages_3phases.png",
    "etape6b_correct_vs_incorrect.png",
    "etape7_evaluation.png",
    "etape7_galerie.png",
]
CAH_IMAGE_ORDER = [
    "cah_etape1_images_brutes.png",
    "cah_etape2_pretraitement.png",
    "cah_etape3_segmentation_tumor.png",
    "cah_etape3_segmentation_sain.png",
    "cah_etape3_segmentation_classes.png",
    "cah_etape6_dendrogramme.png",
    "cah_dendrogramme_reel.png",
    "cah_etape7_nuages_3phases.png",
    "cah_etape7b_correct_incorrect.png",
    "cah_etape8_evaluation.png",
    "cah_etape8_galerie.png",
    "cah_visualisation_clusters_pca.png",
    "cah_galerie_clusters.png",
]
CLASS_LABELS = {
    "glioma_tumor": "Glioma Tumor",
    "meningioma_tumor": "Meningioma Tumor",
    "no_tumor": "No Tumor",
    "pituitary_tumor": "Pituitary Tumor",
}
CAH_CLUSTER_LABELS = {
    0: "Cluster 0",
    1: "Cluster 1",
    2: "Cluster 2",
    3: "Cluster 3",
}
KMEANS_CLASS_DESCRIPTIONS = {
    "glioma_tumor": "Cas avec tumeur de type glioma utilises dans le pipeline K-Means (k = 4).",
    "meningioma_tumor": "Cas avec tumeur de type meningioma utilises dans le pipeline K-Means (k = 4).",
    "no_tumor": "Cas sains sans tumeur, utiles pour comparer la segmentation et verifier l'absence de region tumorale.",
    "pituitary_tumor": "Cas avec tumeur pituitaire utilises dans le pipeline K-Means (k = 4).",
}
BINARY_CLASS_LABELS = {
    "tumor": "Tumor",
    "no_tumor": "No Tumor",
}
BINARY_CLASS_DESCRIPTIONS = {
    "tumor": "Classe binaire regroupant toutes les IRM ou une tumeur est detectee.",
    "no_tumor": "Classe binaire correspondant aux IRM sans tumeur visible.",
}


for directory in [GENERATED_DIR, UPLOAD_DIR, KMEANS_OUTPUT_DIR, KMEANS_BINARY_OUTPUT_DIR, CAH_OUTPUT_DIR, CACHE_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

for directory in [DATA_DEMO_OUTPUT_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


def as_static_path(path: Path) -> str:
    """Convert an absolute asset path into a Flask static relative path."""
    return path.relative_to(STATIC_DIR).as_posix()


def ensure_runtime_paths():
    """Validate that the shared multiclass resources exist before execution."""
    missing = []
    for label, path in {
        "Training": TRAIN_DIR,
        "Testing": TEST_DIR,
        "Script K-Means": KMEANS_SCRIPT,
        "Script CAH": CAH_SCRIPT,
    }.items():
        if not path.exists():
            missing.append(f"{label}: {path}")
    if missing:
        raise FileNotFoundError("Elements manquants:\n" + "\n".join(missing))


def ensure_binary_kmeans_paths():
    """Validate resources needed by the binary K-Means experience."""
    missing = []
    for label, path in {
        "Training binaire": BINARY_TRAIN_DIR,
        "Testing binaire": BINARY_TEST_DIR,
        "Script K-Means binaire": KMEANS_BINARY_SCRIPT,
    }.items():
        if not path.exists():
            missing.append(f"{label}: {path}")
    if missing:
        raise FileNotFoundError("Elements manquants:\n" + "\n".join(missing))


def class_counts(base_dir: Path):
    counts = []
    for class_name in CLASSES:
        class_dir = base_dir / class_name
        total = 0
        if class_dir.exists():
            total = sum(1 for item in class_dir.iterdir() if item.is_file())
        counts.append({"name": class_name, "count": total})
    return counts


def parse_kmeans_accuracy(log_text: str):
    match = re.search(r"Accuracy globale\s*:\s*([0-9.]+)%", log_text)
    if match:
        return float(match.group(1))
    match = re.search(r"accuracy=\s*([0-9.]+)%", log_text, flags=re.IGNORECASE)
    if match:
        return float(match.group(1))
    return None


def filter_kmeans_binary_log(log_text: str) -> str:
    kept_lines = []
    for raw_line in log_text.splitlines():
        line = raw_line.strip()
        if not line:
            if kept_lines and kept_lines[-1] != "":
                kept_lines.append("")
            continue

        lower = line.lower()

        if (
            "analyse images" in lower
            or lower.startswith("...")
            or "pipeline termin" in lower
            or "fichiers génér" in lower
            or "sauvegard" in lower
        ):
            continue

        if any(token in line for token in ["ETAPE", "ÉTAPE", "Ã‰TAPE"]):
            if kept_lines and kept_lines[-1] != "":
                kept_lines.append("")
            kept_lines.append(line)
            continue

        if "Classification MRI" in line or "Segmentation avanc" in line:
            kept_lines.append(line)
            continue

        if "Convergence" in line or "inertie=" in lower:
            kept_lines.append(line)
            continue

        if "images Ã—" in line or "images ×" in line:
            kept_lines.append(line)
            continue

        if "Accuracy globale" in line:
            kept_lines.append(line)
            continue

        if (
            line.startswith("Classe")
            or line.startswith("tumor")
            or line.startswith("no_tumor")
            or line.startswith("Cluster K-Means")
            or line.startswith("Cluster 0")
            or line.startswith("Cluster 1")
            or set(line) <= {"-", "â", "”", "€", "¢"}
        ):
            kept_lines.append(line)

    while kept_lines and kept_lines[-1] == "":
        kept_lines.pop()
    return "\n".join(kept_lines)


def binary_class_counts(base_dir: Path):
    tumor_total = 0
    no_tumor_total = 0
    for class_dir in base_dir.iterdir():
        if not class_dir.is_dir():
            continue
        count = sum(1 for item in class_dir.iterdir() if item.is_file())
        if "no_tumor" in class_dir.name.lower():
            no_tumor_total += count
        else:
            tumor_total += count
    return [
        {"name": "tumor", "count": tumor_total},
        {"name": "no_tumor", "count": no_tumor_total},
    ]


@lru_cache(maxsize=1)
def load_kmeans_module():
    import importlib.util

    spec = importlib.util.spec_from_file_location("brain_kmeans_source", KMEANS_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sorted_image_files(class_dir: Path):
    files = []
    for pattern in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
        files.extend(class_dir.glob(pattern))
    return sorted(files, key=lambda path: path.name.lower())


def _save_image_grid(images, output_path: Path, title: str, cmap=None):
    cols = 5
    rows = int(np.ceil(len(images) / cols)) if images else 1
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.4, rows * 2.4))
    fig.patch.set_facecolor("#0b1220")
    axes = np.array(axes).reshape(rows, cols)
    fig.suptitle(title, fontsize=15, fontweight="bold", color="white")

    for ax in axes.flat:
        ax.axis("off")
        ax.set_facecolor("#111827")

    for ax, image in zip(axes.flat, images):
        if getattr(image, "ndim", 0) == 3:
            ax.imshow(image)
        else:
            ax.imshow(image, cmap=cmap or "gray")
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()


@lru_cache(maxsize=8)
def build_kmeans_class_bundle(class_name: str):
    ensure_runtime_paths()
    if class_name not in CLASSES:
        raise ValueError(f"Classe inconnue: {class_name}")

    run_kmeans_pipeline(force_rerun=False)
    module = load_kmeans_module()
    class_dir = TRAIN_DIR / class_name
    selected_files = _sorted_image_files(class_dir)[: getattr(module, "N_SAMPLES", 50)]
    if not selected_files:
        raise FileNotFoundError(f"Aucune image trouvee pour {class_name}")

    output_dir = KMEANS_OUTPUT_DIR / "classes" / class_name
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_sheet = output_dir / "raw_gallery.png"
    processed_sheet = output_dir / "processed_gallery.png"
    segmented_sheet = output_dir / "segmented_gallery.png"
    sheets_ready = raw_sheet.exists() and processed_sheet.exists() and segmented_sheet.exists()

    if not sheets_ready:
        raw_arrays = []
        processed_arrays = []
        segmented_arrays = []

        for image_path in selected_files:
            raw = Image.open(image_path).convert("L").resize((module.IMG_SIZE, module.IMG_SIZE))
            raw_np = np.array(raw, dtype=np.uint8)
            processed = module.preprocess(raw_np)
            _, _, _, final = module.segment_tumor_full(processed)
            overlay = module.overlay_tumor(processed, final)

            raw_arrays.append(raw_np)
            processed_arrays.append(processed)
            segmented_arrays.append(overlay)

        _save_image_grid(raw_arrays, raw_sheet, f"{CLASS_LABELS[class_name]} - Avant traitement", cmap="gray")
        _save_image_grid(
            processed_arrays,
            processed_sheet,
            f"{CLASS_LABELS[class_name]} - Apres pretraitement",
            cmap="inferno",
        )
        _save_image_grid(
            segmented_arrays,
            segmented_sheet,
            f"{CLASS_LABELS[class_name]} - Segmentation / overlay",
        )

    global_bundle = run_kmeans_pipeline(force_rerun=False)
    focus_files = [
        "etape5a_coude.png",
        "etape6_nuages_3phases.png",
        "etape7_evaluation.png",
        "etape7_galerie.png",
    ]
    if class_name == "no_tumor":
        focus_files.insert(0, "etape3_segmentation_sain.png")
    else:
        focus_files.insert(0, "etape3_segmentation_tumor.png")

    focus_images = [img for img in global_bundle["images"] if img["file_name"] in focus_files]
    focus_images.sort(key=lambda item: focus_files.index(item["file_name"]))

    return {
        "class_name": class_name,
        "class_label": CLASS_LABELS[class_name],
        "description": KMEANS_CLASS_DESCRIPTIONS[class_name],
        "sample_count": len(selected_files),
        "train_total": next(item["count"] for item in class_counts(TRAIN_DIR) if item["name"] == class_name),
        "sheets": [
            {
                "title": "Avant traitement",
                "caption": "Les images brutes utilisees par CAMU avant tout pretraitement.",
                "static_path": as_static_path(raw_sheet),
            },
            {
                "title": "Apres pretraitement",
                "caption": "Le code CAMU applique un flou gaussien, une egalisation d'histogramme puis une normalisation.",
                "static_path": as_static_path(processed_sheet),
            },
            {
                "title": "Segmentation / detection",
                "caption": "Overlay du masque de segmentation sur les images preprocesses.",
                "static_path": as_static_path(segmented_sheet),
            },
        ],
        "focus_images": focus_images,
        "accuracy": global_bundle["accuracy"],
    }


def _save_data_demo_image(image, title: str, output_path: Path, cmap=None):
    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    fig.patch.set_facecolor("#07111f")
    ax.set_facecolor("#07111f")
    if getattr(image, "ndim", 0) == 3:
        ax.imshow(image)
    else:
        ax.imshow(image, cmap=cmap or "gray")
    ax.set_title(title, color="white", fontsize=12, fontweight="bold", pad=10)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()


def _compute_single_feature_vector(module, processed_image):
    flat = processed_image.flatten()
    mean = flat.mean()
    std = flat.std() + 1e-8
    skew = ((flat - mean) ** 3).mean() / std**3
    kurt = ((flat - mean) ** 4).mean() / std**4 - 3
    hist, _ = np.histogram(flat, 32, (0, 1))
    p = hist / (hist.sum() + 1e-8)
    entropy = -np.sum(p * np.log2(p + 1e-10))

    _, _, _, final = module.segment_tumor_full(processed_image)
    t_area = final.sum() / final.size
    t_mean = processed_image[final].mean() if final.sum() > 0 else 0.0

    perimeter = 0
    if final.sum() > 0:
        pad = np.pad(final, 1)
        for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            shifted = pad[1 + di : 1 + di + module.IMG_SIZE, 1 + dj : 1 + dj + module.IMG_SIZE]
            perimeter += np.sum(final & ~shifted)
    compact = perimeter**2 / (final.sum() + 1e-8)

    gy = processed_image[1:, :] - processed_image[:-1, :]
    gx = processed_image[:, 1:] - processed_image[:, :-1]
    grad = (np.abs(gy).mean() + np.abs(gx).mean()) / 2
    homo = 1 - np.abs(processed_image[1:, 1:] - processed_image[:-1, :-1]).mean()

    return [
        ("Intensite moyenne", float(mean)),
        ("Ecart-type", float(std)),
        ("Skewness", float(skew)),
        ("Kurtosis", float(kurt)),
        ("Entropie", float(entropy)),
        ("Surface tumeur", float(t_area)),
        ("Intensite tumeur", float(t_mean)),
        ("Compacite", float(compact)),
        ("Gradient", float(grad)),
        ("Homogeneite", float(homo)),
    ]


@lru_cache(maxsize=1)
def build_data_demo_bundle():
    ensure_runtime_paths()
    module = load_kmeans_module()
    samples = []
    for sample_class in CLASSES:
        sample_dir = TRAIN_DIR / sample_class
        sample_files = _sorted_image_files(sample_dir)
        if not sample_files:
            continue

        sample_path = sample_files[0]
        raw = Image.open(sample_path).convert("L").resize((module.IMG_SIZE, module.IMG_SIZE))
        raw_np = np.array(raw, dtype=np.uint8)
        blurred = np.clip(module.conv2d(raw_np, module.GK), 0, 255).astype(np.uint8)
        equalized = module.equalize_hist(blurred)
        normalized = module.normalize(equalized)
        binary, eroded, conn, final_mask = module.segment_tumor_full(normalized)
        overlay = module.overlay_tumor(normalized, final_mask)

        demo_steps = [
            ("Etape 1 - Image brute", raw_np, "gray", f"{sample_class}_image_brute.png"),
            ("Etape 2 - Flou gaussien", blurred, "gray", f"{sample_class}_image_flou.png"),
            ("Etape 3 - Egalisation", equalized, "gray", f"{sample_class}_image_equalisee.png"),
            ("Etape 4 - Normalisation", normalized, "inferno", f"{sample_class}_image_normalisee.png"),
            ("Etape 5 - Seuillage Otsu", binary.astype(float), "hot", f"{sample_class}_image_otsu.png"),
            ("Etape 6 - Erosion", eroded.astype(float), "hot", f"{sample_class}_image_erosion.png"),
            ("Etape 7 - Connexe principale", conn.astype(float), "hot", f"{sample_class}_image_connexe.png"),
            ("Etape 8 - Detection finale", overlay, None, f"{sample_class}_image_finale.png"),
        ]

        images = []
        for title, image, cmap, file_name in demo_steps:
            output_path = DATA_DEMO_OUTPUT_DIR / file_name
            if not output_path.exists():
                _save_data_demo_image(image, title, output_path, cmap=cmap)
            images.append(
                {
                    "title": title,
                    "static_path": as_static_path(output_path),
                }
            )

        feature_rows = [
            {"name": name, "value": f"{value:.4f}"}
            for name, value in _compute_single_feature_vector(module, normalized)
        ]

        samples.append(
            {
                "sample_class": sample_class,
                "sample_label": CLASS_LABELS.get(sample_class, sample_class),
                "sample_file": sample_path.name,
                "images": images,
                "features": feature_rows,
            }
        )

    if not samples:
        raise FileNotFoundError("Aucune image de demonstration n'a ete trouvee.")

    feature_rows = []
    cached_payload = None
    if DATA_DEMO_FEATURES_PATH.exists():
        try:
            cached_payload = json.loads(DATA_DEMO_FEATURES_PATH.read_text(encoding="utf-8"))
        except Exception:
            cached_payload = None

    if cached_payload and cached_payload.get("rows"):
        feature_rows = cached_payload["rows"]
    else:
        for sample_class in CLASSES:
            sample_dir = TRAIN_DIR / sample_class
            files = _sorted_image_files(sample_dir)
            if hasattr(module, "select_best_images"):
                try:
                    files = module.select_best_images([str(path) for path in files], module.N_SAMPLES)
                    files = [Path(path) for path in files]
                except Exception:
                    files = files[: getattr(module, "N_SAMPLES", 50)]
            else:
                files = files[: getattr(module, "N_SAMPLES", 50)]

            for row_index, sample_path in enumerate(files, start=1):
                raw = Image.open(sample_path).convert("L").resize((module.IMG_SIZE, module.IMG_SIZE))
                raw_np = np.array(raw, dtype=np.uint8)
                normalized = module.preprocess(raw_np)
                values = dict(_compute_single_feature_vector(module, normalized))
                feature_rows.append(
                    {
                        "index": len(feature_rows) + 1,
                        "file_name": sample_path.name,
                        "class_label": CLASS_LABELS.get(sample_class, sample_class),
                        "mean": f"{values['Intensite moyenne']:.4f}",
                        "std": f"{values['Ecart-type']:.4f}",
                        "skew": f"{values['Skewness']:.4f}",
                        "kurt": f"{values['Kurtosis']:.4f}",
                        "entropy": f"{values['Entropie']:.4f}",
                        "tumor_area": f"{values['Surface tumeur']:.4f}",
                        "tumor_mean": f"{values['Intensite tumeur']:.4f}",
                        "compactness": f"{values['Compacite']:.4f}",
                        "gradient": f"{values['Gradient']:.4f}",
                        "homogeneity": f"{values['Homogeneite']:.4f}",
                    }
                )

        DATA_DEMO_FEATURES_PATH.write_text(json.dumps({"rows": feature_rows}, indent=2), encoding="utf-8")

    return {
        "description": (
            "Exemples complets de traitement des images MRI pour chaque type de donnee: "
            "image brute, pretraitement, normalisation, segmentation et extraction des features."
        ),
        "samples": samples,
        "feature_columns": [
            {"key": "index", "label": "#"},
            {"key": "file_name", "label": "Fichier"},
            {"key": "class_label", "label": "Classe"},
            {"key": "mean", "label": "Intensite"},
            {"key": "std", "label": "Std"},
            {"key": "skew", "label": "Skew"},
            {"key": "kurt", "label": "Kurtosis"},
            {"key": "entropy", "label": "Entropie"},
            {"key": "tumor_area", "label": "Surface"},
            {"key": "tumor_mean", "label": "Intensite T."},
            {"key": "compactness", "label": "Compacite"},
            {"key": "gradient", "label": "Gradient"},
            {"key": "homogeneity", "label": "Homogeneite"},
        ],
        "feature_rows": feature_rows,
    }


@lru_cache(maxsize=2)
def run_kmeans_pipeline(force_rerun: bool = False):
    ensure_runtime_paths()
    log_path = KMEANS_OUTPUT_DIR / "kmeans_stdout.txt"
    results_ready = all((KMEANS_OUTPUT_DIR / name).exists() for name in KMEANS_IMAGE_ORDER)

    if force_rerun or not results_ready:
        command = (
            "import importlib.util, os, pathlib, matplotlib; "
            "matplotlib.use('Agg'); "
            f"script_path = r'{KMEANS_SCRIPT}'; "
            f"dataset_path = r'{TRAIN_DIR}'; "
            f"output_dir = pathlib.Path(r'{KMEANS_OUTPUT_DIR}'); "
            "output_dir.mkdir(parents=True, exist_ok=True); "
            "os.chdir(output_dir); "
            "spec = importlib.util.spec_from_file_location('brain_kmeans_module', script_path); "
            "module = importlib.util.module_from_spec(spec); "
            "spec.loader.exec_module(module); "
            "module.main(dataset_path)"
        )
        env = os.environ.copy()
        env["MPLBACKEND"] = "Agg"
        env["PYTHONIOENCODING"] = "utf-8"
        result = subprocess.run(
            [sys.executable, "-X", "utf8", "-c", command],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
            cwd=str(BASE_DIR),
            check=False,
        )
        log_text = (result.stdout or "") + ("\n" + result.stderr if result.stderr else "")
        log_path.write_text(log_text, encoding="utf-8")
    else:
        log_text = log_path.read_text(encoding="utf-8") if log_path.exists() else ""

    images = []
    for file_name in KMEANS_IMAGE_ORDER:
        image_path = KMEANS_OUTPUT_DIR / file_name
        if image_path.exists():
            images.append(
                {
                    "title": file_name.replace(".png", "").replace("_", " ").title(),
                    "file_name": file_name,
                    "static_path": as_static_path(image_path),
                }
            )

    return {
        "description": (
            "Execution du script CAMU/K-Means original sans modification, avec "
            "generation automatique des figures et des resultats."
        ),
        "classes": CLASSES,
        "train_counts": class_counts(TRAIN_DIR),
        "test_counts": class_counts(TEST_DIR),
        "images": images,
        "log_text": log_text.strip(),
        "accuracy": parse_kmeans_accuracy(log_text),
    }


def _binary_counts_for_runtime():
    return {
        "train_counts": binary_class_counts(BINARY_TRAIN_DIR),
        "test_counts": binary_class_counts(BINARY_TEST_DIR),
    }


def _load_kmeans_binary_module():
    import importlib.util

    spec = importlib.util.spec_from_file_location("brain_kmeans_binary_source", KMEANS_BINARY_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_KMEANS_BINARY_RUNTIME = None


def _save_kmeans_binary_runtime_cache(runtime):
    np.savez(
        KMEANS_BINARY_RUNTIME_CACHE,
        centroids=runtime["km"].centroids,
        labels=runtime["km"].labels_,
        mu=runtime["mu"],
        sig=runtime["sig"],
        labels_global=runtime["module"].labels_global,
    )


def _load_kmeans_binary_runtime_from_cache(module):
    if not KMEANS_BINARY_RUNTIME_CACHE.exists():
        return None

    payload = np.load(KMEANS_BINARY_RUNTIME_CACHE, allow_pickle=True)
    km = module.KMeans(
        k=module.N_CLUSTERS,
        max_iter=module.KMEANS_ITER,
        n_init=getattr(module, "KMEANS_NINIT", 1),
    )
    km.centroids = payload["centroids"]
    km.labels_ = payload["labels"]
    module.labels_global = payload["labels_global"]

    log_path = KMEANS_BINARY_OUTPUT_DIR / "kmeans_binary_stdout.txt"
    return {
        "module": module,
        "km": km,
        "mu": payload["mu"],
        "sig": payload["sig"],
        "log_text": log_path.read_text(encoding="utf-8") if log_path.exists() else "",
    }


def _build_lightweight_kmeans_binary_runtime(module):
    previous_cwd = Path.cwd()
    buffer = StringIO()
    try:
        os.chdir(KMEANS_BINARY_OUTPUT_DIR)
        with redirect_stdout(buffer):
            images_raw, labels = module.load_images(str(BINARY_TRAIN_DIR))
            images_proc = module.preprocess_all(images_raw, labels)
            features, valid_indices = module.extract_features(images_proc, labels)
            if len(features) == 0:
                raise RuntimeError("Aucune feature valide n'a pu etre extraite pour K-Means k=2.")

            labels_valid = labels[valid_indices]
            images_proc_valid = images_proc[valid_indices]
            X_norm, mu, sig = module.zscore(features)
            km = module.KMeans(
                k=module.N_CLUSTERS,
                max_iter=module.KMEANS_ITER,
                n_init=getattr(module, "KMEANS_NINIT", 1),
            )
            km.fit(X_norm)
            module.labels_global = labels_valid
            module.images_proc_global = images_proc_valid
    finally:
        os.chdir(previous_cwd)

    return {
        "module": module,
        "km": km,
        "mu": mu,
        "sig": sig,
        "log_text": buffer.getvalue().strip(),
    }


def get_kmeans_binary_runtime(force_rerun: bool = False):
    global _KMEANS_BINARY_RUNTIME

    ensure_binary_kmeans_paths()
    if _KMEANS_BINARY_RUNTIME is not None and not force_rerun:
        return _KMEANS_BINARY_RUNTIME

    output_dir = KMEANS_BINARY_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    module = _load_kmeans_binary_module()

    if not force_rerun:
        cached_runtime = _load_kmeans_binary_runtime_from_cache(module)
        if cached_runtime is not None:
            _KMEANS_BINARY_RUNTIME = cached_runtime
            return _KMEANS_BINARY_RUNTIME

    runtime = _build_lightweight_kmeans_binary_runtime(module)
    log_text = runtime["log_text"]
    (output_dir / "kmeans_binary_stdout.txt").write_text(log_text, encoding="utf-8")
    _save_kmeans_binary_runtime_cache(runtime)

    _KMEANS_BINARY_RUNTIME = {
        "module": runtime["module"],
        "km": runtime["km"],
        "mu": runtime["mu"],
        "sig": runtime["sig"],
        "log_text": log_text.strip(),
    }
    return _KMEANS_BINARY_RUNTIME


@lru_cache(maxsize=2)
def get_kmeans_binary_artifacts(force_rerun: bool = False):
    output_dir = KMEANS_BINARY_OUTPUT_DIR
    log_path = output_dir / "kmeans_binary_stdout.txt"
    results_ready = any((output_dir / name).exists() for name in KMEANS_BINARY_IMAGE_ORDER)

    if force_rerun or not results_ready:
        runtime = get_kmeans_binary_runtime(force_rerun=force_rerun)
        log_text = runtime["log_text"]
    else:
        log_text = log_path.read_text(encoding="utf-8") if log_path.exists() else ""

    images = []
    for file_name in KMEANS_BINARY_IMAGE_ORDER:
        image_path = KMEANS_BINARY_OUTPUT_DIR / file_name
        if image_path.exists():
            images.append(
                {
                    "title": file_name.replace(".png", "").replace("_", " ").title(),
                    "file_name": file_name,
                    "static_path": as_static_path(image_path),
                }
            )

    counts = _binary_counts_for_runtime()
    filtered_log_text = filter_kmeans_binary_log(log_text)
    return {
        "description": (
            "Variante K-Means binaire (k = 2) basee sur le script fourni, avec "
            "classification finale en deux cas: tumeur ou non tumeur."
        ),
        "classes": ["tumor", "no_tumor"],
        "train_counts": counts["train_counts"],
        "test_counts": counts["test_counts"],
        "images": images,
        "log_text": filtered_log_text,
        "accuracy": parse_kmeans_accuracy(log_text),
        "binary_classes": [
            {
                "name": "tumor",
                "label": "Classe 1",
                "train_count": next((item["count"] for item in counts["train_counts"] if item["name"] == "tumor"), 0),
                "test_count": next((item["count"] for item in counts["test_counts"] if item["name"] == "tumor"), 0),
            },
            {
                "name": "no_tumor",
                "label": "Classe 2",
                "train_count": next((item["count"] for item in counts["train_counts"] if item["name"] == "no_tumor"), 0),
                "test_count": next((item["count"] for item in counts["test_counts"] if item["name"] == "no_tumor"), 0),
            },
        ],
    }


@lru_cache(maxsize=2)
def build_kmeans_binary_class_bundle(class_name: str):
    if class_name not in BINARY_CLASS_LABELS:
        raise ValueError(f"Classe binaire inconnue: {class_name}")

    bundle = get_kmeans_binary_artifacts(force_rerun=False)
    image_names = ["etape2_pretraitement.png"]
    if class_name == "tumor":
        image_names.append("etape3_segmentation_tumor.png")
    else:
        image_names.append("etape3_segmentation_sain.png")
    image_names.extend(["etape3_segmentation_classes.png", "etape7_evaluation.png"])

    focus_images = [image for image in bundle["images"] if image["file_name"] in image_names]
    focus_images.sort(key=lambda item: image_names.index(item["file_name"]))

    return {
        "class_name": class_name,
        "class_label": BINARY_CLASS_LABELS[class_name],
        "description": BINARY_CLASS_DESCRIPTIONS[class_name],
        "accuracy": bundle["accuracy"],
        "focus_images": focus_images,
        "interpretation": (
            "Cette classe represente toutes les IRM pour lesquelles le pipeline binaire conclut a la presence d'une tumeur."
            if class_name == "tumor"
            else "Cette classe represente les IRM pour lesquelles le pipeline binaire conclut a l'absence de tumeur."
        ),
    }


def predict_kmeans_binary_image(file: FileStorage):
    runtime = get_kmeans_binary_runtime(force_rerun=False)
    saved_file = _save_uploaded_file(file)

    module = runtime["module"]
    with Image.open(saved_file).convert("L") as pil_image:
        raw = np.array(pil_image.resize((module.IMG_SIZE, module.IMG_SIZE)), dtype=np.uint8)
    processed = module.preprocess(raw)
    final_mask = module.segment_tumor_chanvese(processed)
    overlay = module.overlay_tumor(processed, final_mask)

    feat_vec, _ = module.extract_features(np.array([processed]), np.array([0]))
    if len(feat_vec) == 0:
        raise ValueError("Segmentation insuffisante pour etablir une prediction fiable.")

    feat_norm = (feat_vec - runtime["mu"]) / (runtime["sig"] + 1e-8)
    distance_values = np.linalg.norm(feat_norm - runtime["km"].centroids, axis=1).astype(float)
    cluster_id = int(np.argmin(distance_values))
    _, best_map = module.align_clusters(runtime["km"].labels_, module.labels_global)
    predicted_class = module.CLASSES[best_map[cluster_id]]

    inverse = 1.0 / (distance_values + 1e-8)
    scores = inverse / inverse.sum()
    class_names = [module.CLASSES[best_map[idx]] for idx in range(len(distance_values))]
    probabilities = [
        {
            "class_name": class_name,
            "value": float(score),
            "percent": round(float(score) * 100, 2),
            "distance": round(float(distance), 4),
        }
        for class_name, score, distance in zip(class_names, scores, distance_values)
    ]
    probabilities.sort(key=lambda item: item["value"], reverse=True)

    preview_path = KMEANS_BINARY_OUTPUT_DIR / f"prediction_clean_{saved_file.stem}.png"
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.8))
    fig.patch.set_facecolor("#0D0D0D")
    fig.suptitle(
        f"Prediction K-Means k=2 - {predicted_class}",
        fontsize=13,
        fontweight="bold",
        color="white",
    )
    panels = [
        (raw, "gray", "Image brute"),
        (processed, "gray", "Pre-traitee"),
        (final_mask.astype(float), "hot", "Masque tumeur"),
        (overlay, None, f"Detection - {predicted_class}"),
    ]
    for ax, (data, cmap, title) in zip(axes, panels):
        if cmap is None:
            ax.imshow(data)
        else:
            ax.imshow(data, cmap=cmap)
        ax.set_title(title, color="white", fontsize=10, fontweight="bold", pad=6)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(preview_path, dpi=140, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()

    chart_path = KMEANS_BINARY_OUTPUT_DIR / f"prediction_scores_{saved_file.stem}.png"
    ordered_names = [item["class_name"] for item in probabilities]
    ordered_scores = [item["value"] for item in probabilities]
    plt.figure(figsize=(7.2, 4.5))
    bars = plt.barh(ordered_names, ordered_scores, color=["#2563eb", "#ef4444"][: len(ordered_scores)])
    plt.xlim(0, 1)
    plt.xlabel("Score")
    plt.title("Distances / scores par cluster")
    for bar, item in zip(bars, probabilities):
        plt.text(
            bar.get_width() + 0.015,
            bar.get_y() + bar.get_height() / 2,
            f"{item['percent']}%",
            va="center",
        )
    plt.tight_layout()
    plt.savefig(chart_path, dpi=140)
    plt.close()

    return {
        "predicted_class": predicted_class,
        "cluster_id": cluster_id,
        "uploaded_image": as_static_path(saved_file),
        "prediction_image": as_static_path(preview_path),
        "chart_image": as_static_path(chart_path),
        "probabilities": probabilities,
    }


@lru_cache(maxsize=2)
def run_cah_pipeline(force_rerun: bool = False):
    ensure_runtime_paths()
    log_path = CAH_OUTPUT_DIR / "cah_stdout.txt"
    results_ready = any((CAH_OUTPUT_DIR / name).exists() for name in CAH_IMAGE_ORDER)

    if force_rerun or not results_ready:
        command = (
            "import importlib.util, os, pathlib, matplotlib; "
            "matplotlib.use('Agg'); "
            f"script_path = r'{CAH_SCRIPT}'; "
            f"dataset_path = r'{TRAIN_DIR}'; "
            f"output_dir = pathlib.Path(r'{CAH_OUTPUT_DIR}'); "
            "output_dir.mkdir(parents=True, exist_ok=True); "
            "os.chdir(output_dir); "
            "spec = importlib.util.spec_from_file_location('brain_cah_module', script_path); "
            "module = importlib.util.module_from_spec(spec); "
            "spec.loader.exec_module(module); "
            f"module.CLASSES[:] = {CLASSES!r}; "
            "module.main(dataset_path)"
        )
        env = os.environ.copy()
        env["MPLBACKEND"] = "Agg"
        env["PYTHONIOENCODING"] = "utf-8"
        result = subprocess.run(
            [sys.executable, "-X", "utf8", "-c", command],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
            cwd=str(BASE_DIR),
            check=False,
        )
        log_text = (result.stdout or "") + ("\n" + result.stderr if result.stderr else "")
        log_path.write_text(log_text, encoding="utf-8")
    else:
        log_text = log_path.read_text(encoding="utf-8") if log_path.exists() else ""

    images = []
    for file_name in CAH_IMAGE_ORDER:
        image_path = CAH_OUTPUT_DIR / file_name
        if image_path.exists():
            images.append(
                {
                    "title": file_name.replace(".png", "").replace("_", " ").title(),
                    "file_name": file_name,
                    "static_path": as_static_path(image_path),
                }
            )

    return {
        "description": (
            "Execution du script CAH original sans modification, avec generation du "
            "dendrogramme, des nuages de points, de l'evaluation et des sorties visuelles."
        ),
        "classes": CLASSES,
        "train_counts": class_counts(TRAIN_DIR),
        "test_counts": class_counts(TEST_DIR),
        "images": images,
        "log_text": log_text.strip(),
        "accuracy": parse_kmeans_accuracy(log_text),
    }


_CAH_RUNTIME = None


def _save_cah_summary(payload):
    CAH_SUMMARY_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _load_cah_summary():
    if not CAH_SUMMARY_PATH.exists():
        return None
    return json.loads(CAH_SUMMARY_PATH.read_text(encoding="utf-8"))


def _build_cah_cluster_cards_from_runtime(runtime):
    labels = runtime["labels"]
    cah_labels = runtime["cluster_labels"]
    cards = []

    for cluster_id in range(runtime["cluster_count"]):
        idxs = np.where(cah_labels == cluster_id)[0]
        mapped_class = runtime["cluster_majority"][cluster_id]
        distribution = []
        for class_id, class_name in enumerate(CLASSES):
            count = int(np.sum(labels[idxs] == class_id))
            if count:
                distribution.append({"name": class_name, "count": count})
        cards.append(
            {
                "cluster_id": cluster_id,
                "label": f"Cluster {cluster_id + 1}",
                "count": int(len(idxs)),
                "mapped_class": mapped_class,
                "distribution": distribution,
            }
        )
    return cards


def _load_cah_module():
    import importlib.util

    spec = importlib.util.spec_from_file_location("brain_cah_source", CAH_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.CLASSES[:] = CLASSES
    return module


def _choose_cah_cluster_count(merge_history, min_clusters=2):
    n_samples = len(merge_history) + 1
    if n_samples <= 1:
        return 1

    heights = np.array([row[0] for row in merge_history], dtype=float)
    if len(heights) <= 1:
        return min(min_clusters, n_samples)

    diffs = np.diff(heights)
    ranked = np.argsort(diffs)[::-1]

    for idx in ranked:
        candidate_k = n_samples - (idx + 1)
        if min_clusters <= candidate_k < n_samples:
            return int(candidate_k)

    return min(min_clusters, n_samples)


def _labels_from_cah_history(merge_history, n_samples, target_clusters):
    clusters = {i: [i] for i in range(n_samples)}

    for _, left_idx, right_idx, new_idx, _, _ in merge_history:
        if len(clusters) <= target_clusters:
            break
        merged = clusters.pop(left_idx) + clusters.pop(right_idx)
        clusters[new_idx] = merged

    labels = np.zeros(n_samples, dtype=int)
    ordered_clusters = sorted(clusters.items(), key=lambda item: item[0])
    for cluster_label, (_, members) in enumerate(ordered_clusters):
        labels[members] = cluster_label

    return labels


def _dominant_class_name(labels, idxs):
    if len(idxs) == 0:
        return "indetermine"

    counts = []
    for class_id, class_name in enumerate(CLASSES):
        count = int(np.sum(labels[idxs] == class_id))
        counts.append((count, class_name))
    counts.sort(reverse=True)
    return counts[0][1]


def get_cah_runtime(force_rerun: bool = False):
    global _CAH_RUNTIME
    ensure_runtime_paths()
    if _CAH_RUNTIME is not None and not force_rerun:
        return _CAH_RUNTIME

    module = _load_cah_module()
    previous_cwd = Path.cwd()
    buffer = StringIO()
    try:
        os.chdir(CAH_OUTPUT_DIR)
        with redirect_stdout(buffer):
            images_raw, labels = module.load_images(str(TRAIN_DIR))
            images_proc = module.preprocess_all(images_raw, labels)
            features = module.extract_features(images_proc, labels)
            x_norm = module.zscore(features)
            cah = module.CAH(n_clusters=1, linkage="ward")
            cah.fit(x_norm)
    finally:
        os.chdir(previous_cwd)

    n_samples = len(labels)
    auto_cluster_count = _choose_cah_cluster_count(cah.merge_history, min_clusters=2)
    auto_labels = _labels_from_cah_history(cah.merge_history, n_samples, auto_cluster_count)
    purity_total = 0
    dominant_classes = {}
    for cluster_id in range(auto_cluster_count):
        idxs = np.where(auto_labels == cluster_id)[0]
        dominant_name = _dominant_class_name(labels, idxs)
        dominant_classes[cluster_id] = dominant_name
        if len(idxs):
            class_counts = [int(np.sum(labels[idxs] == class_id)) for class_id in range(len(CLASSES))]
            purity_total += max(class_counts)

    _CAH_RUNTIME = {
        "module": module,
        "images_raw": images_raw,
        "images_proc": images_proc,
        "labels": labels,
        "features": features,
        "x_norm": x_norm,
        "cah": cah,
        "cluster_count": auto_cluster_count,
        "cluster_labels": auto_labels,
        "cluster_majority": dominant_classes,
        "purity": (purity_total / n_samples) if n_samples else 0.0,
        "log_text": buffer.getvalue().strip(),
    }
    _save_cah_summary(
        {
            "cluster_count": _CAH_RUNTIME["cluster_count"],
            "purity_percent": _CAH_RUNTIME["purity"] * 100,
            "decision_method": "Coupe automatique du dendrogramme par saut maximal de distance",
            "cluster_cards": _build_cah_cluster_cards_from_runtime(_CAH_RUNTIME),
        }
    )
    return _CAH_RUNTIME


def _build_cah_cluster_cards():
    summary = _load_cah_summary()
    if summary and summary.get("cluster_cards"):
        return summary["cluster_cards"]
    runtime = get_cah_runtime(force_rerun=False)
    cards = _build_cah_cluster_cards_from_runtime(runtime)
    for index, card in enumerate(cards, start=1):
        card["label"] = f"Cluster {index}"
    return cards


def get_cah_cluster_cards():
    return _build_cah_cluster_cards()


def get_cah_cluster_overview():
    summary = _load_cah_summary()
    if summary:
        return {
            "cluster_count": summary.get("cluster_count"),
            "purity_percent": summary.get("purity_percent"),
            "decision_method": summary.get("decision_method"),
        }
    runtime = get_cah_runtime(force_rerun=False)
    return {
        "cluster_count": runtime["cluster_count"],
        "purity_percent": runtime["purity"] * 100,
        "decision_method": "Coupe automatique du dendrogramme par saut maximal de distance",
    }


def build_cah_cluster_bundle(cluster_id: int):
    runtime = get_cah_runtime(force_rerun=False)
    if cluster_id < 0 or cluster_id >= runtime["cluster_count"]:
        raise ValueError(f"Cluster CAH inconnu: {cluster_id}")

    images_proc = runtime["images_proc"]
    labels = runtime["labels"]
    cah_labels = runtime["cluster_labels"]
    idxs = np.where(cah_labels == cluster_id)[0]
    if len(idxs) == 0:
        raise ValueError("Aucune image dans ce cluster.")

    cluster_dir = CAH_OUTPUT_DIR / "clusters" / f"cluster_{cluster_id}"
    cluster_dir.mkdir(parents=True, exist_ok=True)

    sample_idxs = idxs[: min(20, len(idxs))]
    gallery_images = []
    for order, img_idx in enumerate(sample_idxs, start=1):
        image_path = cluster_dir / f"cluster_{cluster_id}_{order}.png"
        _save_data_demo_image(
            images_proc[img_idx],
            f"Image {order}",
            image_path,
            cmap="gray",
        )
        gallery_images.append(
            {
                "title": f"Image {order} - {CLASSES[int(labels[img_idx])]}",
                "static_path": as_static_path(image_path),
            }
        )

    distribution = []
    for class_id, class_name in enumerate(CLASSES):
        count = int(np.sum(labels[idxs] == class_id))
        distribution.append({"name": class_name, "count": count})

    focus_files = [
        "cah_etape6_dendrogramme.png",
        "cah_dendrogramme_reel.png",
        "cah_etape7_nuages_3phases.png",
        "cah_etape8_evaluation.png",
        "cah_etape8_galerie.png",
    ]
    focus_images = []
    for file_name in focus_files:
        image_path = CAH_OUTPUT_DIR / file_name
        if image_path.exists():
            focus_images.append(
                {
                    "title": file_name.replace(".png", "").replace("_", " ").title(),
                    "static_path": as_static_path(image_path),
                }
            )

    return {
        "cluster_id": cluster_id,
        "cluster_label": f"Cluster {cluster_id + 1}",
        "mapped_class": runtime["cluster_majority"][cluster_id],
        "sample_count": int(len(idxs)),
        "distribution": distribution,
        "gallery_images": gallery_images,
        "focus_images": focus_images,
        "purity": runtime["purity"] * 100,
        "cluster_count": runtime["cluster_count"],
        "description": "Images regroupees par la coupe automatique du dendrogramme CAH, avec la repartition des vraies classes.",
    }



def _save_uploaded_file(file: FileStorage) -> Path:
    allowed = {".png", ".jpg", ".jpeg", ".bmp", ".jfif", ".webp"}
    extension = Path(file.filename or "").suffix.lower()
    if extension not in allowed:
        raise ValueError("Formats acceptes: PNG, JPG, JPEG, BMP, JFIF, WEBP.")

    safe_name = secure_filename(file.filename or "upload.png")
    timestamp_name = f"{int(time.time())}_{safe_name}"
    output_path = UPLOAD_DIR / timestamp_name
    file.save(output_path)
    return output_path
