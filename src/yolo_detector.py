"""
YOLOv8-based object detector.
Approche Deep Learning : détection multi-classes avec un CNN pré-entraîné sur COCO.
"""

import cv2
import numpy as np
from typing import List, Optional, Tuple
from ultralytics import YOLO


class YOLODetector:
    """
    Détecteur d'objets basé sur YOLOv8 (Deep Learning).
    
    Avantages : détecte 80 classes d'objets, robuste à l'éclairage, précision élevée.
    Limites : plus lent que HSV (~15-30 FPS sur CPU), nécessite ~6 Mo de modèle.
    """

    # Classes COCO disponibles (extrait des plus courantes)
    COCO_CLASSES_FR = {
        "person": "personne", "bicycle": "vélo", "car": "voiture",
        "motorcycle": "moto", "bus": "bus", "cat": "chat", "dog": "chien",
        "bottle": "bouteille", "cup": "tasse", "chair": "chaise",
        "laptop": "ordinateur", "mouse": "souris", "keyboard": "clavier",
        "cell phone": "téléphone", "book": "livre", "scissors": "ciseaux",
    }

    def __init__(self, model_size: str = "n", confidence_threshold: float = 0.5):
        """
        Args:
            model_size: Taille du modèle YOLO ('n'=nano, 's'=small, 'm'=medium)
                       'n' = le plus rapide, idéal pour CPU
            confidence_threshold: Seuil de confiance minimum (0.0 à 1.0)
        """
        model_name = f"yolov8{model_size}.pt"
        print(f"Chargement du modèle {model_name}...")
        self.model = YOLO(model_name)
        self.confidence_threshold = confidence_threshold
        print("Modèle chargé.")

    def detect(self, frame: np.ndarray) -> List[dict]:
        """
        Détecte tous les objets dans une frame.
        
        Args:
            frame: Image au format BGR (numpy array)
        
        Returns:
            Liste de dicts : [{'bbox': (x1,y1,x2,y2), 'label': str, 'confidence': float}, ...]
        """
        results = self.model(frame, verbose=False)[0]
        detections = []

        for box in results.boxes:
            conf = float(box.conf[0])
            if conf < self.confidence_threshold:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            class_id = int(box.cls[0])
            label = self.model.names[class_id]

            detections.append({
                "bbox": (x1, y1, x2, y2),
                "label": label,
                "confidence": conf,
            })

        return detections

    def annotate(self, frame: np.ndarray, detections: List[dict]) -> np.ndarray:
        """
        Dessine les bounding boxes et labels sur la frame.
        """
        annotated = frame.copy()

        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            label = det["label"]
            conf = det["confidence"]

            # Bbox bleue pour YOLO (différent du HSV qui est vert)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 100, 0), 2)

            # Label avec fond pour la lisibilité
            text = f"{label} {conf:.0%}"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(annotated, (x1, y1 - th - 10), (x1 + tw + 4, y1), (255, 100, 0), -1)
            cv2.putText(annotated, text, (x1 + 2, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        return annotated


# Test rapide si on lance ce fichier directement
if __name__ == "__main__":
    print("Test du YOLODetector avec la webcam...")
    print("YOLO détecte 80 classes : personne, téléphone, bouteille, chaise...")
    print("Appuie sur 'q' pour quitter.\n")

    detector = YOLODetector(model_size="n", confidence_threshold=0.5)
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detections = detector.detect(frame)
        annotated = detector.annotate(frame, detections)

        # Afficher le nombre d'objets détectés en haut à gauche
        info = f"Objets detectes: {len(detections)}"
        cv2.putText(annotated, info, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("YOLOv8 Detector - Press Q to quit", annotated)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Fini !")