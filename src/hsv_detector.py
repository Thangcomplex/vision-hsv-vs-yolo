import cv2
import numpy as np 
from typing import Optional,Tuple, List


class HSVDetector:
    def __init__(self, target_color_bgr: List[int], hue_tolerance: int = 10):
        self.target_color_bgr = target_color_bgr
        self.hue_tolerance = hue_tolerance
        self.lower_limit, self.upper_limit = self._compute_hsv_limits()

    def _compute_hsv_limits(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calcule les bornes HSV à partir de la couleur BGR cible.
        Gère le wrap-around du rouge (hue proche de 0 ou 180).
        """
        c = np.uint8([[self.target_color_bgr]])
        hsv_c = cv2.cvtColor(c, cv2.COLOR_BGR2HSV)
        hue = int(hsv_c[0][0][0])

        if hue >= 180 - self.hue_tolerance - 5:  # Rouge haut
            lower = np.array([hue - self.hue_tolerance, 100, 100], dtype=np.uint8)
            upper = np.array([180, 255, 255], dtype=np.uint8)
        elif hue <= self.hue_tolerance + 5:  # Rouge bas
            lower = np.array([0, 100, 100], dtype=np.uint8)
            upper = np.array([hue + self.hue_tolerance, 255, 255], dtype=np.uint8)
        else:
            lower = np.array([hue - self.hue_tolerance, 100, 100], dtype=np.uint8)
            upper = np.array([hue + self.hue_tolerance, 255, 255], dtype=np.uint8)

        return lower, upper

    def detect(self, frame: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """
        Détecte la couleur cible dans une frame.
        
        Args:
            frame: Image au format BGR (numpy array)
        
        Returns:
            Bounding box (x1, y1, x2, y2) ou None si rien détecté.
        """
        hsv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv_image, self.lower_limit, self.upper_limit)

        # Trouver les contours plutôt qu'une simple bbox (plus robuste)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return None

        # Garder le plus gros contour (l'objet principal)
        largest = max(contours, key=cv2.contourArea)
        
        # Filtrer le bruit (très petits contours)
        if cv2.contourArea(largest) < 500:
            return None

        x, y, w, h = cv2.boundingRect(largest)
        return (x, y, x + w, y + h)

    def annotate(self, frame: np.ndarray, bbox: Optional[Tuple[int, int, int, int]]) -> np.ndarray:
        """
        Dessine la bounding box sur la frame.
        """
        annotated = frame.copy()
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.putText(annotated, "HSV Detection", (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        return annotated


# Test rapide si on lance ce fichier directement
if __name__ == "__main__":
    print("Test du HSVDetector avec la webcam...")
    print("Appuie sur 'q' pour quitter.\n")

    detector = HSVDetector(target_color_bgr=[0, 255, 255])  # Jaune
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        bbox = detector.detect(frame)
        annotated = detector.annotate(frame, bbox)

        cv2.imshow("HSV Detector - Press Q to quit", annotated)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Fini !")