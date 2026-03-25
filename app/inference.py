"""
inference.py
============
Charge le modèle BiLSTM entraîné et prédit le signe à partir
d'une séquence de keypoints reçue via WebSocket.

Responsabilités :
  - Charger action.h5 + labels.npy une seule fois au démarrage
  - Accumuler les frames dans un buffer glissant
  - Prédire quand le buffer est plein (sequence_length frames)
  - Retourner le label + la confidence + le top-3 des prédictions
"""

import numpy as np
import os
from pathlib import Path
from collections import deque
from typing import Optional

# TensorFlow/Keras — chargement différé pour éviter les imports lents au démarrage
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"   # Masque les warnings CUDA/GPU

import tensorflow as tf
from tensorflow import keras

# ─── Constantes ───────────────────────────────────────────────────────────────

MODELS_DIR      = Path(__file__).parent.parent / "ai_engine" / "models"
MODEL_PATH      = MODELS_DIR / "action.h5"
LABELS_PATH     = MODELS_DIR / "labels.npy"

# Doit correspondre à ce qui a été utilisé dans data_processor.py
SEQUENCE_LENGTH = 30
KEYPOINTS_DIM   = 1662

# Seuil minimum de confiance pour retourner une prédiction (évite les faux positifs)
CONFIDENCE_THRESHOLD = 0.70


# ─── Classe principale ────────────────────────────────────────────────────────

class EchoSignInference:
    """
    Moteur d'inférence pour EchoSign.
    Instancié une seule fois dans main.py (singleton).
    """

    def __init__(
        self,
        model_path: str      = str(MODEL_PATH),
        labels_path: str     = str(LABELS_PATH),
        sequence_length: int = SEQUENCE_LENGTH,
        confidence_threshold: float = CONFIDENCE_THRESHOLD,
    ):
        self.sequence_length      = sequence_length
        self.confidence_threshold = confidence_threshold
        self.model: Optional[keras.Model] = None
        self.labels: Optional[np.ndarray] = None

        # Buffer glissant : garde les N dernières frames reçues
        # Permet une prédiction continue sans attendre une séquence complète
        self.frame_buffer = deque(maxlen=sequence_length)

        self._load(model_path, labels_path)

    def _load(self, model_path: str, labels_path: str):
        """Charge le modèle et les labels depuis le disque."""
        print(f"[Inference] Chargement du modèle depuis {model_path} ...")

        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"Modèle introuvable : {model_path}\n"
                f"Lance d'abord : python ai_engine/train_model.py"
            )
        if not Path(labels_path).exists():
            raise FileNotFoundError(
                f"Labels introuvables : {labels_path}\n"
                f"Lance d'abord : python ai_engine/data_processor.py"
            )

        self.model  = keras.models.load_model(model_path)
        self.labels = np.load(labels_path, allow_pickle=True)

        print(f"[Inference] ✓ Modèle chargé — {len(self.labels)} signes : {list(self.labels)}")
        print(f"[Inference] ✓ Séquence attendue : {self.sequence_length} frames × {KEYPOINTS_DIM} keypoints")

    # ── API principale ─────────────────────────────────────────────────────────

    def add_frame(self, keypoints: list | np.ndarray) -> dict:
        """
        Ajoute une frame (1662 keypoints) au buffer et retourne une prédiction.

        Args:
            keypoints: liste ou array de 1662 valeurs float (une frame MediaPipe)

        Returns:
            dict avec :
              - "label"       : str  — signe prédit ("BONJOUR")
              - "confidence"  : float — confiance 0.0→1.0
              - "top3"        : list  — top 3 [{"label": str, "score": float}]
              - "ready"       : bool  — False si le buffer n'est pas encore plein
              - "buffer_fill" : int   — frames actuellement dans le buffer
        """
        kp = np.array(keypoints, dtype=np.float32)

        # Validation de la forme
        if kp.shape != (KEYPOINTS_DIM,):
            return {
                "error": f"Keypoints invalides: attendu ({KEYPOINTS_DIM},), reçu {kp.shape}",
                "ready": False,
                "buffer_fill": len(self.frame_buffer),
            }

        self.frame_buffer.append(kp)

        # Pas encore assez de frames pour prédire
        if len(self.frame_buffer) < self.sequence_length:
            return {
                "label": None,
                "confidence": 0.0,
                "top3": [],
                "ready": False,
                "buffer_fill": len(self.frame_buffer),
            }

        # Buffer plein → on prédit
        return self._predict()

    def predict_sequence(self, sequence: list | np.ndarray) -> dict:
        """
        Prédit directement depuis une séquence complète.
        Utile pour l'endpoint REST (envoi d'un bloc de frames d'un coup).

        Args:
            sequence: array de shape (sequence_length, 1662)

        Returns:
            dict identique à add_frame()
        """
        seq = np.array(sequence, dtype=np.float32)
        if seq.shape != (self.sequence_length, KEYPOINTS_DIM):
            return {
                "error": (
                    f"Séquence invalide: attendu ({self.sequence_length}, {KEYPOINTS_DIM}), "
                    f"reçu {seq.shape}"
                ),
                "ready": False,
            }

        # Remplace le buffer par cette séquence
        self.frame_buffer.clear()
        for frame in seq:
            self.frame_buffer.append(frame)

        return self._predict()

    # ── Prédiction interne ─────────────────────────────────────────────────────

    def _predict(self) -> dict:
        """Effectue la prédiction sur le buffer courant."""
        # Construire l'input du modèle: (1, sequence_length, 1662)
        sequence = np.array(self.frame_buffer, dtype=np.float32)
        input_tensor = np.expand_dims(sequence, axis=0)  # (1, 30, 1662)

        # Inférence
        predictions = self.model.predict(input_tensor, verbose=0)[0]  # (nb_classes,)

        # Classe la plus probable
        best_idx   = int(np.argmax(predictions))
        best_conf  = float(predictions[best_idx])
        best_label = str(self.labels[best_idx])

        # Top-3 prédictions
        top3_idx = np.argsort(predictions)[::-1][:3]
        top3 = [
            {"label": str(self.labels[i]), "score": round(float(predictions[i]), 4)}
            for i in top3_idx
        ]

        # Si confiance insuffisante → on ne retourne pas de signe
        if best_conf < self.confidence_threshold:
            return {
                "label": None,
                "confidence": round(best_conf, 4),
                "top3": top3,
                "ready": True,
                "buffer_fill": len(self.frame_buffer),
                "below_threshold": True,
            }

        return {
            "label": best_label,
            "confidence": round(best_conf, 4),
            "top3": top3,
            "ready": True,
            "buffer_fill": len(self.frame_buffer),
            "below_threshold": False,
        }

    def reset_buffer(self):
        """Vide le buffer (à appeler entre deux mots/phrases)."""
        self.frame_buffer.clear()

    @property
    def is_loaded(self) -> bool:
        return self.model is not None and self.labels is not None

    @property
    def sign_count(self) -> int:
        return len(self.labels) if self.labels is not None else 0

    @property
    def sign_list(self) -> list:
        return list(self.labels) if self.labels is not None else []


# ─── Instance partagée (importée dans main.py) ────────────────────────────────
# Instanciée une seule fois au démarrage du serveur

_engine: Optional[EchoSignInference] = None


def get_engine() -> EchoSignInference:
    """Retourne le singleton du moteur d'inférence."""
    global _engine
    if _engine is None:
        _engine = EchoSignInference()
    return _engine


def init_engine(**kwargs) -> EchoSignInference:
    """Initialise explicitement le moteur avec des paramètres personnalisés."""
    global _engine
    _engine = EchoSignInference(**kwargs)
    return _engine