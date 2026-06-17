# 🎯 Détection d'objets : Vision Classique vs Deep Learning

> Benchmark scientifique comparant une approche de vision classique (segmentation HSV) et un modèle Deep Learning moderne (YOLOv8), avec interface interactive Streamlit.

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-yellow.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-FF4B4B.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

---

## 📋 Description

Ce projet compare deux approches radicalement différentes de **détection d'objets** dans une image :

- **HSV (Vision Classique)** : segmentation par couleur via OpenCV — rapide mais limitée à une seule classe
- **YOLOv8 (Deep Learning)** : CNN pré-entraîné sur COCO (80 classes) — plus lent mais beaucoup plus polyvalent et robuste

Le but : quantifier scientifiquement les compromis **vitesse vs robustesse vs polyvalence**.

## 🎯 Composants

### 1. Application interactive (`app.py`)
Interface Streamlit permettant de :
- Uploader une image ou prendre une photo (webcam)
- Régler la couleur cible HSV et le seuil de confiance YOLO
- Comparer les détections côte à côte avec métriques (temps, nombre d'objets)

### 2. Benchmark scientifique (`benchmark.py`)
Pipeline complet d'évaluation :
- Phase de **warm-up** (élimination du biais du premier appel)
- Mesure des temps d'inférence (moyenne ± écart-type)
- Comptage des détections et taux de détection
- Génération automatique de rapports **CSV** et **graphiques matplotlib**

### 3. Détecteurs modulaires (`src/`)
Architecture orientée objet :
- `HSVDetector` : segmentation par seuillage HSV avec gestion du wrap-around du rouge
- `YOLODetector` : wrapper autour d'Ultralytics YOLOv8 avec annotations OpenCV

## 📊 Résultats typiques

| Métrique | HSV | YOLOv8 |
|---|---|---|
| Temps moyen | ~2 ms | ~50 ms |
| FPS théorique | ~500 | ~20 |
| Classes détectées | 1 (couleur) | 80 (COCO) |
| Robustesse luminosité | ⚠️ Faible | ✅ Élevée |
| Taux de détection (jaune) | ~XX% | ~XX% |

> 📝 Remplacez les XX par les chiffres réels après votre benchmark

![Benchmark comparison](results/benchmark_comparison.png)

## 🚀 Installation

```bash
# Cloner le dépôt
git clone https://github.com/Thangcomplex/vision-hsv-vs-yolo.git
cd vision-hsv-vs-yolo

# Créer un environnement virtuel
python -m venv .venv
.venv\Scripts\activate    # Windows
# source .venv/bin/activate  # Linux/Mac

# Installer les dépendances
pip install -r requirements.txt
```

## 💻 Utilisation

### Lancer l'application interactive

```bash
streamlit run app.py
```

→ Ouvre automatiquement votre navigateur sur `http://localhost:8501`

### Lancer le benchmark

1. Placez quelques images de test dans `data/test_images/`
2. Lancez :
```bash
python benchmark.py
```
3. Les résultats apparaissent dans `results/` :
   - `benchmark_detailed.csv` — résultats par image
   - `benchmark_summary.csv` — statistiques agrégées
   - `benchmark_comparison.png` — graphiques

### Détection de couleur en temps réel (webcam)

```bash
python Detect_Color.py
```

Détecte la couleur jaune en direct depuis votre webcam (appuyez sur `q` pour quitter).

## 📁 Structure

```
.
├── app.py                  # Application Streamlit interactive
├── benchmark.py            # Pipeline de benchmark scientifique
├── Detect_Color.py         # Démo webcam temps réel
├── src/
│   ├── hsv_detector.py     # Classe HSVDetector
│   └── yolo_detector.py    # Classe YOLODetector
├── data/
│   └── test_images/        # Images pour le benchmark
├── results/                # Sorties générées (CSV, PNG)
├── yolov8n.pt              # Poids pré-entraînés YOLOv8 nano
└── requirements.txt
```

## 🔬 Méthodologie scientifique

Le benchmark suit les bonnes pratiques :

1. **Warm-up** : 2 itérations avant la mesure (élimine le coût du premier appel CUDA)
2. **Mesures multiples** : moyenne et écart-type sur N images
3. **Métriques multiples** : temps, FPS théorique, taux de détection
4. **Reproductibilité** : graine fixée, configuration documentée

## 🛠️ Stack technique

- **Python 3.10+**
- **OpenCV** — vision classique (HSV, masking, bbox)
- **Ultralytics YOLOv8** — Deep Learning (CNN pré-entraîné)
- **PyTorch** — backend DL
- **Streamlit** — interface web
- **Matplotlib + NumPy** — analyse et visualisation

## 🎓 Enseignements

Ce projet illustre que **« moderne » n'est pas toujours synonyme de « meilleur »** :

- Pour détecter **une seule couleur fixe** sur un système embarqué temps réel → HSV reste imbattable (~500 FPS)
- Pour de la **détection multi-classes robuste** en environnement variable → YOLOv8 s'impose
- La **bonne approche** dépend du cahier des charges, pas de la hype technologique

## 📄 Licence

MIT

## 👤 Auteur

**Pham Duc Thang** — Étudiant ingénieur, INSA Centre-Val de Loire
- GitHub : [@Thangcomplex](https://github.com/Thangcomplex)
- Email : votre.email@example.com
