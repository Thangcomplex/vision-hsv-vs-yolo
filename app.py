"""
Application Streamlit — Détection d'objets : HSV vs YOLOv8.
Permet de comparer une approche classique (segmentation HSV) et 
un modèle Deep Learning (YOLOv8) sur des images ou un flux webcam.
"""

import time
import cv2
import numpy as np
import streamlit as st
from PIL import Image

from src.hsv_detector import HSVDetector
from src.yolo_detector import YOLODetector


# ----------------------------------------------------------------------------
# Configuration de la page
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Vision Classique vs Deep Learning",
    page_icon="🎯",
    layout="wide",
)

st.title("🎯 Détection d'objets : Vision Classique vs Deep Learning")
st.markdown(
    """
    Comparaison de deux approches de computer vision sur la même image :
    - **HSV** (vision classique) : segmentation par couleur, rapide mais limitée
    - **YOLOv8** (Deep Learning) : CNN pré-entraîné, multi-classes, plus lent
    """
)


# ----------------------------------------------------------------------------
# Chargement des modèles (mis en cache pour éviter de recharger à chaque fois)
# ----------------------------------------------------------------------------
@st.cache_resource
def load_yolo():
    return YOLODetector(model_size="n", confidence_threshold=0.5)


# ----------------------------------------------------------------------------
# Sidebar : paramètres
# ----------------------------------------------------------------------------
st.sidebar.header("⚙️ Paramètres")

mode = st.sidebar.radio(
    "Source d'entrée",
    ["📤 Upload d'image", "📷 Webcam (snapshot)"],
)

st.sidebar.markdown("---")
st.sidebar.subheader("HSV — couleur cible")
color_preset = st.sidebar.selectbox(
    "Préréglage",
    ["Jaune", "Bleu", "Rouge", "Vert", "Personnalisé"],
)
color_map = {
    "Jaune": [0, 255, 255],
    "Bleu": [255, 0, 0],
    "Rouge": [0, 0, 255],
    "Vert": [0, 255, 0],
}
if color_preset == "Personnalisé":
    b = st.sidebar.slider("B", 0, 255, 0)
    g = st.sidebar.slider("G", 0, 255, 255)
    r = st.sidebar.slider("R", 0, 255, 255)
    target_color = [b, g, r]
else:
    target_color = color_map[color_preset]

hue_tol = st.sidebar.slider("Tolérance de teinte HSV", 5, 30, 10)

st.sidebar.markdown("---")
st.sidebar.subheader("YOLOv8")
yolo_conf = st.sidebar.slider("Seuil de confiance", 0.1, 0.9, 0.5)


# ----------------------------------------------------------------------------
# Récupération de l'image selon le mode
# ----------------------------------------------------------------------------
frame = None

if mode == "📤 Upload d'image":
    uploaded = st.file_uploader(
        "Choisis une image (JPG, PNG)",
        type=["jpg", "jpeg", "png"],
    )
    if uploaded is not None:
        pil_img = Image.open(uploaded).convert("RGB")
        frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

elif mode == "📷 Webcam (snapshot)":
    snapshot = st.camera_input("Prends une photo")
    if snapshot is not None:
        pil_img = Image.open(snapshot).convert("RGB")
        frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


# ----------------------------------------------------------------------------
# Détection et affichage
# ----------------------------------------------------------------------------
if frame is not None:
    # --- HSV ---
    hsv_detector = HSVDetector(target_color_bgr=target_color, hue_tolerance=hue_tol)
    t0 = time.perf_counter()
    hsv_bbox = hsv_detector.detect(frame)
    hsv_time_ms = (time.perf_counter() - t0) * 1000
    hsv_annotated = hsv_detector.annotate(frame, hsv_bbox)

    # --- YOLO ---
    yolo_detector = load_yolo()
    yolo_detector.confidence_threshold = yolo_conf
    t0 = time.perf_counter()
    yolo_detections = yolo_detector.detect(frame)
    yolo_time_ms = (time.perf_counter() - t0) * 1000
    yolo_annotated = yolo_detector.annotate(frame, yolo_detections)

    # --- Affichage côte à côte ---
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🎨 HSV (Vision Classique)")
        st.image(cv2.cvtColor(hsv_annotated, cv2.COLOR_BGR2RGB), use_container_width=True)
        st.metric("Temps d'inférence", f"{hsv_time_ms:.1f} ms")
        st.metric("Objets détectés", 1 if hsv_bbox else 0)

    with col2:
        st.subheader("🧠 YOLOv8 (Deep Learning)")
        st.image(cv2.cvtColor(yolo_annotated, cv2.COLOR_BGR2RGB), use_container_width=True)
        st.metric("Temps d'inférence", f"{yolo_time_ms:.1f} ms")
        st.metric("Objets détectés", len(yolo_detections))

    # --- Tableau récapitulatif ---
    st.markdown("---")
    st.subheader("📊 Comparaison")

    speedup = yolo_time_ms / hsv_time_ms if hsv_time_ms > 0 else 0

    comparison_data = {
        "Méthode": ["HSV (classique)", "YOLOv8 (Deep Learning)"],
        "Temps (ms)": [f"{hsv_time_ms:.1f}", f"{yolo_time_ms:.1f}"],
        "FPS théorique": [f"{1000/hsv_time_ms:.0f}", f"{1000/yolo_time_ms:.0f}"],
        "Objets détectés": [1 if hsv_bbox else 0, len(yolo_detections)],
        "Multi-classes": ["❌ (1 couleur)", "✅ (80 classes)"],
        "Robustesse lumière": ["⚠️ Faible", "✅ Élevée"],
    }
    st.table(comparison_data)

    if speedup > 1:
        st.info(f"💡 HSV est **{speedup:.1f}× plus rapide** que YOLO sur cette image, "
                f"mais YOLO détecte plus d'objets et est plus robuste.")

    # --- Détails YOLO ---
    if yolo_detections:
        with st.expander("🔍 Détails des détections YOLO"):
            for i, det in enumerate(yolo_detections, 1):
                st.write(f"**{i}. {det['label']}** — confiance : {det['confidence']:.1%} — bbox : {det['bbox']}")

else:
    st.info("👆 Upload une image ou prends une photo pour démarrer.")