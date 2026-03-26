"""
data_processor.py  (mediapipe 0.10.30+ Tasks API)
==================================================
Convertit tes vidéos (.mp4, .avi, .mov) ET tes images (.jpg, .png)
en séquences numpy prêtes pour l'entraînement BiLSTM.

Structure attendue dans raw_data/ :
    raw_data/
    ├── BONJOUR/
    │   ├── video1.mp4
    │   └── photo1.jpg
    ├── MERCI/
    │   └── merci_clip.mp4
    └── A/
        └── lettre_a.png

Résultat dans dataset/ :
    dataset/
    ├── BONJOUR/
    │   ├── 0.npy    ← (sequence_length, 1692)
    │   └── 1.npy
    └── ...

Usage :
    python ai_engine/data_processor.py
    python ai_engine/data_processor.py --sequence_length 45
    python ai_engine/data_processor.py --raw_data custom_raw/ --dataset custom_out/
"""

import cv2
import numpy as np
import os
import argparse
import urllib.request
from pathlib import Path

# ─── Téléchargement automatique des modèles MediaPipe ────────────────────────

_MODELS_DIR = Path(__file__).parent.parent / "mp_models"
_MODELS_DIR.mkdir(exist_ok=True)

_MODEL_URLS = {
    "hand_landmarker.task": (
        "https://storage.googleapis.com/mediapipe-models/"
        "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    ),
    "pose_landmarker.task": (
        "https://storage.googleapis.com/mediapipe-models/"
        "pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
    ),
    "face_landmarker.task": (
        "https://storage.googleapis.com/mediapipe-models/"
        "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
    ),
}

def _ensure_models():
    for fname, url in _MODEL_URLS.items():
        dest = _MODELS_DIR / fname
        if not dest.exists():
            print(f"  ⬇  Téléchargement {fname} ...")
            urllib.request.urlretrieve(url, str(dest))
            print(f"  ✓  {fname} prêt.")

_ensure_models()

# ─── Import Tasks API ─────────────────────────────────────────────────────────

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode

# ─── Constantes ───────────────────────────────────────────────────────────────

# pose(33×4) + face(478×3) + main_g(21×3) + main_d(21×3) = 1692
KEYPOINTS_PER_FRAME  = 1692
DEFAULT_SEQUENCE_LENGTH = 30

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# ─── Détecteur ────────────────────────────────────────────────────────────────

class _Detector:
    """Détecteur combiné Pose + Face + Hands (Tasks API)."""

    def __init__(self, detection_conf=0.5, tracking_conf=0.5):
        Base = mp_python.BaseOptions
        self._pose = mp_vision.PoseLandmarker.create_from_options(
            mp_vision.PoseLandmarkerOptions(
                base_options=Base(model_asset_path=str(_MODELS_DIR / "pose_landmarker.task")),
                running_mode=VisionTaskRunningMode.IMAGE,
                min_pose_detection_confidence=detection_conf,
                min_tracking_confidence=tracking_conf,
            )
        )
        self._face = mp_vision.FaceLandmarker.create_from_options(
            mp_vision.FaceLandmarkerOptions(
                base_options=Base(model_asset_path=str(_MODELS_DIR / "face_landmarker.task")),
                running_mode=VisionTaskRunningMode.IMAGE,
                min_face_detection_confidence=detection_conf,
                min_tracking_confidence=tracking_conf,
                num_faces=1,
            )
        )
        self._hands = mp_vision.HandLandmarker.create_from_options(
            mp_vision.HandLandmarkerOptions(
                base_options=Base(model_asset_path=str(_MODELS_DIR / "hand_landmarker.task")),
                running_mode=VisionTaskRunningMode.IMAGE,
                min_hand_detection_confidence=detection_conf,
                min_tracking_confidence=tracking_conf,
                num_hands=2,
            )
        )

    def detect(self, frame_bgr):
        """Retourne (pose_lms, face_lms, left_hand_lms, right_hand_lms)."""
        rgb    = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        pose_res  = self._pose.detect(mp_img)
        face_res  = self._face.detect(mp_img)
        hand_res  = self._hands.detect(mp_img)

        pose_lms = pose_res.pose_landmarks[0] if pose_res.pose_landmarks else None
        face_lms = face_res.face_landmarks[0] if face_res.face_landmarks else None

        left_hand = right_hand = None
        if hand_res.hand_landmarks:
            for lm_list, handedness in zip(hand_res.hand_landmarks,
                                           hand_res.handedness):
                label = handedness[0].category_name
                if label == "Left":
                    left_hand  = lm_list
                else:
                    right_hand = lm_list

        return pose_lms, face_lms, left_hand, right_hand

    def close(self):
        self._pose.close()
        self._face.close()
        self._hands.close()

    def __enter__(self): return self
    def __exit__(self, *_): self.close()


# ─── Extraction des keypoints ─────────────────────────────────────────────────

def extract_keypoints(pose_lms, face_lms, left_hand, right_hand) -> np.ndarray:
    """
    Vecteur 1D de 1692 valeurs :
      pose(33×4) + face(478×3) + main_g(21×3) + main_d(21×3)
    """
    # Pose : 33 × 4
    if pose_lms:
        pose = np.array([[lm.x, lm.y, lm.z,
                          getattr(lm, 'visibility', 0.0)]
                         for lm in pose_lms]).flatten()
    else:
        pose = np.zeros(33 * 4)

    # Visage : 478 × 3
    if face_lms:
        raw    = np.array([[lm.x, lm.y, lm.z] for lm in face_lms]).flatten()
        target = 478 * 3
        face   = np.zeros(target)
        face[:min(len(raw), target)] = raw[:target]
    else:
        face = np.zeros(478 * 3)

    # Mains : 21 × 3
    lh = (np.array([[lm.x, lm.y, lm.z] for lm in left_hand]).flatten()
          if left_hand else np.zeros(21 * 3))
    rh = (np.array([[lm.x, lm.y, lm.z] for lm in right_hand]).flatten()
          if right_hand else np.zeros(21 * 3))

    return np.concatenate([pose, face, lh, rh])  # (1692,)


# ─── Traitement d'une vidéo ───────────────────────────────────────────────────

def video_to_sequences(
    video_path: str,
    sequence_length: int,
    detector: _Detector,
    verbose: bool = True,
) -> list:
    """
    Découpe une vidéo en séquences de `sequence_length` frames (fenêtre glissante 50%).
    Si la vidéo est trop courte, la dernière frame est répétée (padding).
    Retourne une liste de np.ndarray (sequence_length, KEYPOINTS_PER_FRAME).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  [ERREUR] Impossible d'ouvrir {video_path}")
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps          = cap.get(cv2.CAP_PROP_FPS) or 30.0

    if verbose:
        print(f"  Vidéo : {Path(video_path).name} | {total_frames} frames | {fps:.1f} fps")

    all_keypoints = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)  # Même orientation que la collecte webcam
        kp = extract_keypoints(*detector.detect(frame))
        all_keypoints.append(kp)

    cap.release()

    if not all_keypoints:
        print(f"  [ATTENTION] Aucune frame lue depuis {video_path}")
        return []

    # Découpage en séquences avec fenêtre glissante (50% overlap)
    sequences = []
    stride = max(1, sequence_length // 2)
    i = 0
    while i + sequence_length <= len(all_keypoints):
        sequences.append(np.array(all_keypoints[i:i + sequence_length]))
        i += stride

    # Vidéo trop courte → padding par répétition de la dernière frame
    if not sequences:
        padded = list(all_keypoints)
        while len(padded) < sequence_length:
            padded.append(padded[-1])
        sequences.append(np.array(padded[:sequence_length]))
        if verbose:
            print(f"  [INFO] Vidéo courte, paddée à {sequence_length} frames.")

    return sequences


# ─── Traitement d'une image ───────────────────────────────────────────────────

def image_to_sequence(
    image_path: str,
    sequence_length: int,
    detector: _Detector,
    verbose: bool = True,
):
    """
    Lit une image et la répète `sequence_length` fois.
    Utile pour les lettres statiques (A, B, C...).
    Retourne np.ndarray (sequence_length, KEYPOINTS_PER_FRAME) ou None.
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"  [ERREUR] Impossible de lire {image_path}")
        return None

    if verbose:
        print(f"  Image : {Path(image_path).name} | répétée {sequence_length}×")

    kp = extract_keypoints(*detector.detect(image))
    return np.tile(kp, (sequence_length, 1))  # (seq_len, 1692)


# ─── Pipeline principal ───────────────────────────────────────────────────────

def process_all(
    raw_data_dir: str     = "raw_data",
    dataset_dir:  str     = "dataset",
    sequence_length: int  = DEFAULT_SEQUENCE_LENGTH,
    verbose: bool         = True,
):
    raw_path     = Path(raw_data_dir)
    dataset_path = Path(dataset_dir)

    if not raw_path.exists():
        print(f"[ERREUR] Dossier introuvable : {raw_data_dir}")
        print("         Crée raw_data/BONJOUR/, raw_data/MERCI/ ...")
        return

    sign_folders = [f for f in raw_path.iterdir() if f.is_dir()]
    if not sign_folders:
        print(f"[ERREUR] Aucun sous-dossier dans {raw_data_dir}")
        return

    print(f"\n{'='*60}")
    print(f"  EchoSign — Génération du dataset (Tasks API)")
    print(f"  Source   : {raw_data_dir}/")
    print(f"  Dest     : {dataset_dir}/")
    print(f"  Séquence : {sequence_length} frames  ({sequence_length/30:.1f}s à 30fps)")
    print(f"  Keypoints: {KEYPOINTS_PER_FRAME} valeurs/frame")
    print(f"{'='*60}\n")

    total_sequences = 0
    stats = {}

    with _Detector() as detector:
        for sign_folder in sorted(sign_folders):
            sign_name = sign_folder.name
            out_dir   = dataset_path / sign_name
            out_dir.mkdir(parents=True, exist_ok=True)

            print(f"▶  Signe : {sign_name}")

            # Reprendre sans écraser l'existant
            seq_index  = len(list(out_dir.glob("*.npy")))
            sign_count = 0

            files = sorted(sign_folder.iterdir())
            if not files:
                print(f"  [ATTENTION] Dossier vide : {sign_folder}\n")
                continue

            for file_path in files:
                suffix = file_path.suffix.lower()

                if suffix in VIDEO_EXTENSIONS:
                    seqs = video_to_sequences(
                        str(file_path), sequence_length, detector, verbose)
                    for seq in seqs:
                        np.save(str(out_dir / f"{seq_index}.npy"), seq)
                        seq_index  += 1
                        sign_count += 1

                elif suffix in IMAGE_EXTENSIONS:
                    seq = image_to_sequence(
                        str(file_path), sequence_length, detector, verbose)
                    if seq is not None:
                        np.save(str(out_dir / f"{seq_index}.npy"), seq)
                        seq_index  += 1
                        sign_count += 1

                else:
                    if verbose:
                        print(f"  [IGNORÉ] {file_path.name}")

            stats[sign_name]     = sign_count
            total_sequences     += sign_count
            print(f"  ✓ {sign_count} séquence(s) → {out_dir}/\n")

    # Résumé
    print(f"\n{'='*60}")
    print(f"  Résumé du dataset")
    print(f"{'='*60}")
    max_count = max(stats.values()) if stats else 1
    for sign, count in sorted(stats.items()):
        bar = "█" * int(count / max(max_count, 1) * 30)
        print(f"  {sign:<20} {bar:<30} {count}")
    print(f"\n  Total : {total_sequences} séquences dans {dataset_dir}/")

    # Sauvegarder les labels
    labels_path = dataset_path / "labels.npy"
    np.save(str(labels_path), np.array(sorted(stats.keys())))
    print(f"  Labels → {labels_path}")
    print(f"{'='*60}\n")


# ─── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="EchoSign — Conversion vidéo/image → .npy (Tasks API)"
    )
    parser.add_argument("--raw_data",        default="raw_data",
                        help="Dossier source (défaut: raw_data)")
    parser.add_argument("--dataset",         default="dataset",
                        help="Dossier de sortie (défaut: dataset)")
    parser.add_argument("--sequence_length", type=int, default=DEFAULT_SEQUENCE_LENGTH,
                        help=f"Frames par séquence (défaut: {DEFAULT_SEQUENCE_LENGTH})")
    parser.add_argument("--quiet",           action="store_true",
                        help="Moins de logs")
    args = parser.parse_args()

    process_all(
        raw_data_dir    = args.raw_data,
        dataset_dir     = args.dataset,
        sequence_length = args.sequence_length,
        verbose         = not args.quiet,
    )