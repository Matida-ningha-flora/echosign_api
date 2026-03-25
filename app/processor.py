"""
processor.py
============
Valide et nettoie les keypoints reçus depuis Flutter avant de les passer
à inference.py.

Flutter peut envoyer soit :
  - Une frame isolée (1662 keypoints) via WebSocket temps réel
  - Une séquence complète (sequence_length × 1662) via REST POST
"""

import numpy as np
from typing import Any

KEYPOINTS_DIM   = 1662
SEQUENCE_LENGTH = 30


class ProcessorError(ValueError):
    """Erreur levée quand les données reçues sont invalides."""
    pass


def clean_frame(raw: Any) -> np.ndarray:
    """
    Valide et normalise une frame de keypoints (1662 valeurs).

    Args:
        raw: liste, dict {"keypoints": [...]}, ou array brut reçu de Flutter

    Returns:
        np.ndarray de shape (1662,) dtype float32

    Raises:
        ProcessorError si les données sont invalides
    """
    # Extraire la liste selon le format reçu
    if isinstance(raw, dict):
        if "keypoints" not in raw:
            raise ProcessorError(
                f"Clé 'keypoints' manquante dans le payload. Reçu : {list(raw.keys())}"
            )
        data = raw["keypoints"]
    elif isinstance(raw, (list, np.ndarray)):
        data = raw
    else:
        raise ProcessorError(f"Format non supporté : {type(raw)}")

    # Convertir en numpy
    try:
        arr = np.array(data, dtype=np.float32)
    except (ValueError, TypeError) as e:
        raise ProcessorError(f"Impossible de convertir en float32 : {e}")

    # Vérification de la forme
    arr = arr.flatten()
    if arr.shape[0] != KEYPOINTS_DIM:
        raise ProcessorError(
            f"Nombre de keypoints incorrect : attendu {KEYPOINTS_DIM}, reçu {arr.shape[0]}"
        )

    # Vérification NaN / Inf
    if not np.isfinite(arr).all():
        nan_count = np.sum(np.isnan(arr))
        inf_count = np.sum(np.isinf(arr))
        # Remplacer par 0 plutôt que rejeter (MediaPipe peut produire des NaN sur landmark absent)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        # Log pour le debug (optionnel)
        if nan_count + inf_count > 50:
            print(f"[Processor] ATTENTION : {nan_count} NaN + {inf_count} Inf remplacés par 0")

    return arr


def clean_sequence(raw: Any) -> np.ndarray:
    """
    Valide une séquence complète (plusieurs frames d'un coup).
    Utilisé par l'endpoint REST /predict/sequence.

    Args:
        raw: liste de listes, ou dict {"sequence": [[...], ...]}

    Returns:
        np.ndarray de shape (SEQUENCE_LENGTH, KEYPOINTS_DIM) dtype float32
    """
    if isinstance(raw, dict):
        if "sequence" not in raw:
            raise ProcessorError(
                f"Clé 'sequence' manquante. Reçu : {list(raw.keys())}"
            )
        data = raw["sequence"]
    elif isinstance(raw, (list, np.ndarray)):
        data = raw
    else:
        raise ProcessorError(f"Format non supporté : {type(raw)}")

    try:
        arr = np.array(data, dtype=np.float32)
    except (ValueError, TypeError) as e:
        raise ProcessorError(f"Impossible de convertir la séquence en float32 : {e}")

    if arr.ndim != 2:
        raise ProcessorError(
            f"Séquence mal formée : attendu 2D (frames × keypoints), reçu {arr.ndim}D"
        )

    frames, kp = arr.shape
    if frames != SEQUENCE_LENGTH:
        raise ProcessorError(
            f"Nombre de frames incorrect : attendu {SEQUENCE_LENGTH}, reçu {frames}"
        )
    if kp != KEYPOINTS_DIM:
        raise ProcessorError(
            f"Dimension keypoints incorrecte : attendu {KEYPOINTS_DIM}, reçu {kp}"
        )

    # Nettoyer les NaN / Inf
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

    return arr