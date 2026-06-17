"""
Benchmark scientifique — HSV vs YOLOv8.
Mesure les performances des deux détecteurs sur un ensemble d'images de test.
Génère un rapport CSV et des graphiques de comparaison.
"""

import time
from pathlib import Path
import csv

import cv2
import numpy as np
import matplotlib.pyplot as plt

from src.hsv_detector import HSVDetector
from src.yolo_detector import YOLODetector


# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
TEST_IMAGES_DIR = Path("data/test_images")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

# Pour HSV, on cible le jaune par défaut (couleur fréquente)
HSV_TARGET_COLOR = [0, 255, 255]  # Jaune en BGR
YOLO_CONFIDENCE = 0.5
N_WARMUP = 2  # Itérations de warm-up (pas comptées dans les métriques)


# ----------------------------------------------------------------------------
# Chargement des images
# ----------------------------------------------------------------------------
def load_test_images() -> list:
    extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    image_paths = sorted([
        p for p in TEST_IMAGES_DIR.iterdir()
        if p.suffix.lower() in extensions
    ])
    
    if not image_paths:
        raise FileNotFoundError(
            f"Aucune image trouvée dans {TEST_IMAGES_DIR}. "
            "Ajoute au moins 5 images avant de lancer le benchmark."
        )
    
    print(f"📁 {len(image_paths)} images trouvées dans {TEST_IMAGES_DIR}")
    return image_paths


# ----------------------------------------------------------------------------
# Benchmark d'un détecteur sur toutes les images
# ----------------------------------------------------------------------------
def benchmark_detector(detector, image_paths: list, method_name: str) -> list:
    print(f"\n🔬 Benchmark de {method_name}...")
    results = []
    
    # Warm-up (le 1er appel d'un modèle est toujours plus lent)
    dummy = cv2.imread(str(image_paths[0]))
    for _ in range(N_WARMUP):
        detector.detect(dummy)
    
    for i, img_path in enumerate(image_paths, 1):
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  ⚠️ Impossible de lire {img_path.name}, skip.")
            continue
        
        h, w = img.shape[:2]
        
        t0 = time.perf_counter()
        detections = detector.detect(img)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        
        # Compter les objets détectés
        if isinstance(detections, list):
            n_objects = len(detections)
        elif detections is None:
            n_objects = 0
        else:
            n_objects = 1
        
        results.append({
            "image": img_path.name,
            "method": method_name,
            "width": w,
            "height": h,
            "time_ms": elapsed_ms,
            "fps_theoretical": 1000 / elapsed_ms if elapsed_ms > 0 else 0,
            "n_objects": n_objects,
        })
        
        print(f"  [{i}/{len(image_paths)}] {img_path.name}: "
              f"{elapsed_ms:.1f} ms, {n_objects} objet(s)")
    
    return results


# ----------------------------------------------------------------------------
# Calcul des statistiques agrégées
# ----------------------------------------------------------------------------
def compute_stats(results: list, method_name: str) -> dict:
    method_results = [r for r in results if r["method"] == method_name]
    times = [r["time_ms"] for r in method_results]
    objects = [r["n_objects"] for r in method_results]
    
    return {
        "method": method_name,
        "n_images": len(method_results),
        "time_mean_ms": np.mean(times),
        "time_std_ms": np.std(times),
        "time_min_ms": np.min(times),
        "time_max_ms": np.max(times),
        "fps_mean": np.mean([1000/t for t in times if t > 0]),
        "objects_mean": np.mean(objects),
        "objects_total": np.sum(objects),
        "detection_rate": np.mean([1 if n > 0 else 0 for n in objects]) * 100,
    }


# ----------------------------------------------------------------------------
# Génération des graphiques
# ----------------------------------------------------------------------------
def plot_results(results: list, stats: list):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    hsv_results = [r for r in results if r["method"] == "HSV"]
    yolo_results = [r for r in results if r["method"] == "YOLOv8"]
    
    # --- Graphique 1 : Temps d'inférence par image ---
    ax = axes[0]
    indices = range(1, len(hsv_results) + 1)
    ax.plot(indices, [r["time_ms"] for r in hsv_results],
            "o-", label="HSV", color="#4CAF50", linewidth=2)
    ax.plot(indices, [r["time_ms"] for r in yolo_results],
            "o-", label="YOLOv8", color="#2196F3", linewidth=2)
    ax.set_xlabel("Image #")
    ax.set_ylabel("Temps d'inférence (ms)")
    ax.set_title("Temps d'inférence par image")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_yscale("log")
    
    # --- Graphique 2 : Objets détectés par image ---
    ax = axes[1]
    x = np.arange(len(hsv_results))
    width = 0.35
    ax.bar(x - width/2, [r["n_objects"] for r in hsv_results],
           width, label="HSV", color="#4CAF50")
    ax.bar(x + width/2, [r["n_objects"] for r in yolo_results],
           width, label="YOLOv8", color="#2196F3")
    ax.set_xlabel("Image #")
    ax.set_ylabel("Nombre d'objets détectés")
    ax.set_title("Objets détectés par image")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    
    # --- Graphique 3 : Comparaison globale (barres) ---
    ax = axes[2]
    methods = [s["method"] for s in stats]
    time_means = [s["time_mean_ms"] for s in stats]
    time_stds = [s["time_std_ms"] for s in stats]
    colors = ["#4CAF50", "#2196F3"]
    
    bars = ax.bar(methods, time_means, yerr=time_stds, color=colors,
                  capsize=10, edgecolor="black", linewidth=1.5)
    ax.set_ylabel("Temps moyen d'inférence (ms)")
    ax.set_title("Comparaison globale (moyenne ± écart-type)")
    ax.grid(alpha=0.3, axis="y")
    
    for bar, val in zip(bars, time_means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f"{val:.1f} ms", ha="center", va="bottom", fontweight="bold")
    
    plt.tight_layout()
    output_path = RESULTS_DIR / "benchmark_comparison.png"
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    print(f"\n📊 Graphique sauvegardé : {output_path}")
    plt.close()


# ----------------------------------------------------------------------------
# Sauvegarde CSV
# ----------------------------------------------------------------------------
def save_csv(results: list, stats: list):
    # Détails par image
    detailed_path = RESULTS_DIR / "benchmark_detailed.csv"
    with open(detailed_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"📄 Détails CSV : {detailed_path}")
    
    # Stats agrégées
    summary_path = RESULTS_DIR / "benchmark_summary.csv"
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=stats[0].keys())
        writer.writeheader()
        writer.writerows(stats)
    print(f"📄 Résumé CSV : {summary_path}")


# ----------------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("  BENCHMARK : Vision Classique (HSV) vs Deep Learning (YOLOv8)")
    print("=" * 70)
    
    # Charger les images
    image_paths = load_test_images()
    
    # Initialiser les détecteurs
    hsv = HSVDetector(target_color_bgr=HSV_TARGET_COLOR, hue_tolerance=10)
    yolo = YOLODetector(model_size="n", confidence_threshold=YOLO_CONFIDENCE)
    
    # Benchmark
    results = []
    results.extend(benchmark_detector(hsv, image_paths, "HSV"))
    results.extend(benchmark_detector(yolo, image_paths, "YOLOv8"))
    
    # Stats
    stats = [
        compute_stats(results, "HSV"),
        compute_stats(results, "YOLOv8"),
    ]
    
    # Affichage du rapport final
    print("\n" + "=" * 70)
    print("  📊 RAPPORT FINAL")
    print("=" * 70)
    
    for s in stats:
        print(f"\n🔹 {s['method']}")
        print(f"   Images traitées       : {s['n_images']}")
        print(f"   Temps moyen           : {s['time_mean_ms']:.2f} ms (±{s['time_std_ms']:.2f})")
        print(f"   Temps min/max         : {s['time_min_ms']:.2f} / {s['time_max_ms']:.2f} ms")
        print(f"   FPS théorique moyen   : {s['fps_mean']:.1f}")
        print(f"   Objets/image (moyenne): {s['objects_mean']:.2f}")
        print(f"   Total objets détectés : {s['objects_total']}")
        print(f"   Taux de détection     : {s['detection_rate']:.1f}%")
    
    # Comparaison
    hsv_time = stats[0]["time_mean_ms"]
    yolo_time = stats[1]["time_mean_ms"]
    hsv_obj = stats[0]["objects_mean"]
    yolo_obj = stats[1]["objects_mean"]
    
    print("\n" + "─" * 70)
    print("  🎯 SYNTHÈSE COMPARATIVE")
    print("─" * 70)
    print(f"  ⚡ HSV est {yolo_time/hsv_time:.1f}× plus rapide que YOLOv8")
    if hsv_obj > 0:
        print(f"  🎯 YOLOv8 détecte {yolo_obj/hsv_obj:.1f}× plus d'objets en moyenne")
    else:
        print(f"  🎯 YOLOv8 détecte {yolo_obj:.1f} objets/image vs 0 pour HSV")
    print()
    
    # Génération sorties
    plot_results(results, stats)
    save_csv(results, stats)
    
    print("\n✅ Benchmark terminé. Résultats dans le dossier 'results/'.")


if __name__ == "__main__":
    main()