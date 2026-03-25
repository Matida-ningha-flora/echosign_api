"""
data_processor.py
=================
Convertit tes vidéos (.mp4, .avi, .mov) ET tes images (.jpg, .png)
en séquences numpy prêtes pour l'entraînement BiLSTM.

Structure attendue dans raw_data/ :
    raw_data/
    ├── BONJOUR/
    │   ├── video1.mp4
    │   ├── video2.avi
    │   └── photo1.jpg        ← une image = une séquence "figée"
    ├── MERCI/
    │   └── merci_clip.mp4
    └── A/
        └── lettre_a.png

Résultat dans dataset/ :
    dataset/
    ├── BONJOUR/
    │   ├── 0.npy             ← séquence 0 (30 frames de keypoints)
    │   ├── 1.npy
    │   └── 2.npy
    ├── MERCI/
    │   └── 0.npy
    └── A/
        └── 0.npy

Usage :
    python ai_engine/data_processor.py
    python ai_engine/data_processor.py --sequence_length 45
    python ai_engine/data_processor.py --raw_data custom_raw/ --dataset custom_dataset/
"""

import cv2
import numpy as np
import mediapipe as mp
import os
import argparse
from pathlib import Path

# ─── Constantes ───────────────────────────────────────────────────────────────

# Nombre de keypoints extraits par MediaPipe Holistic par frame
# Pose: 33×4=132 | Face: 468×3=1404 | Main gauche: 21×3=63 | Main droite: 21×3=63
# Total: 1662 valeurs par frame
KEYPOINTS_PER_FRAME = 1662

# Nombre de frames par séquence (doit correspondre à ce que le modèle attend)
DEFAULT_SEQUENCE_LENGTH = 30

# Extensions acceptées
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# ─── Initialisation MediaPipe ─────────────────────────────────────────────────

mp_holistic   = mp.solutions.holistic
mp_drawing    = mp.solutions.drawing_utils


# ─── Extraction des keypoints ─────────────────────────────────────────────────

def extract_keypoints(results) -> np.ndarray:
    """
    Aplatit tous les landmarks MediaPipe Holistic en un vecteur 1D de 1662 valeurs.
    Si un landmark est absent (main non détectée), remplace par des zéros.
    """
    pose = (
        np.array([[lm.x, lm.y, lm.z, lm.visibility] for lm in results.pose_landmarks.landmark]).flatten()
        if results.pose_landmarks else np.zeros(33 * 4)
    )
    face = (
        np.array([[lm.x, lm.y, lm.z] for lm in results.face_landmarks.landmark]).flatten()
        if results.face_landmarks else np.zeros(468 * 3)
    )
    lh = (
        np.array([[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark]).flatten()
        if results.left_hand_landmarks else np.zeros(21 * 3)
    )
    rh = (
        np.array([[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark]).flatten()
        if results.right_hand_landmarks else np.zeros(21 * 3)
    )
    return np.concatenate([pose, face, lh, rh])  # shape: (1662,)


def mediapipe_detection(image: np.ndarray, model):
    """
    Envoie une frame BGR à MediaPipe et retourne les résultats.
    """
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_rgb.flags.writeable = False
    results = model.process(image_rgb)
    image_rgb.flags.writeable = True
    return results


# ─── Traitement d'une vidéo → liste de séquences ─────────────────────────────

def video_to_sequences(
    video_path: str,
    sequence_length: int,
    holistic,
    verbose: bool = True,
) -> list[np.ndarray]:
    """
    Découpe une vidéo en séquences de `sequence_length` frames.
    Chaque séquence = np.ndarray de shape (sequence_length, 1662).

    Retourne une liste de séquences.
    Si la vidéo est trop courte, on duplique les dernières frames pour compléter.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  [ERREUR] Impossible d'ouvrir {video_path}")
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps          = cap.get(cv2.CAP_PROP_FPS)

    if verbose:
        print(f"  Vidéo: {Path(video_path).name} | {total_frames} frames | {fps:.1f} fps")

    # Extraire tous les keypoints de la vidéo frame par frame
    all_keypoints = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        results = mediapipe_detection(frame, holistic)
        kp = extract_keypoints(results)
        all_keypoints.append(kp)

    cap.release()

    if len(all_keypoints) == 0:
        print(f"  [ATTENTION] Aucune frame lue depuis {video_path}")
        return []

    # Découper en séquences de sequence_length frames
    sequences = []
    i = 0
    while i + sequence_length <= len(all_keypoints):
        seq = np.array(all_keypoints[i : i + sequence_length])  # (seq_len, 1662)
        sequences.append(seq)
        i += sequence_length // 2  # Fenêtre glissante avec 50% overlap

    # Si la vidéo est trop courte pour une séquence complète → on pad
    if len(sequences) == 0:
        padded = all_keypoints.copy()
        while len(padded) < sequence_length:
            padded.append(padded[-1])  # Duplique la dernière frame
        sequences.append(np.array(padded[:sequence_length]))
        print(f"  [INFO] Vidéo courte, paddée à {sequence_length} frames.")

    return sequences


# ─── Traitement d'une image → 1 séquence "figée" ─────────────────────────────

def image_to_sequence(
    image_path: str,
    sequence_length: int,
    holistic,
    verbose: bool = True,
) -> np.ndarray | None:
    """
    Lit une image et la répète `sequence_length` fois pour former une séquence.
    Utile pour les lettres statiques (A, B, C...) de la LSF.
    Retourne np.ndarray de shape (sequence_length, 1662) ou None si échec.
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"  [ERREUR] Impossible de lire l'image {image_path}")
        return None

    if verbose:
        print(f"  Image: {Path(image_path).name} | répétée {sequence_length}x")

    results  = mediapipe_detection(image, holistic)
    kp       = extract_keypoints(results)

    # Répéter le même keypoint pour remplir la séquence
    sequence = np.tile(kp, (sequence_length, 1))  # (seq_len, 1662)
    return sequence


# ─── Pipeline principal ───────────────────────────────────────────────────────

def process_all(
    raw_data_dir: str = "raw_data",
    dataset_dir:  str = "dataset",
    sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
    verbose: bool = True,
):
    """
    Parcourt raw_data/<SIGNE>/<fichier> et génère dataset/<SIGNE>/<n>.npy
    """
    raw_path     = Path(raw_data_dir)
    dataset_path = Path(dataset_dir)

    if not raw_path.exists():
        print(f"[ERREUR] Le dossier raw_data introuvable : {raw_data_dir}")
        print("         Crée-le et ajoute tes vidéos/images dedans.")
        return

    # Liste des signes = sous-dossiers de raw_data/
    sign_folders = [f for f in raw_path.iterdir() if f.is_dir()]
    if not sign_folders:
        print(f"[ERREUR] Aucun sous-dossier dans {raw_data_dir}")
        print("         Structure attendue : raw_data/BONJOUR/, raw_data/MERCI/ ...")
        return

    print(f"\n{'='*60}")
    print(f"  EchoSign — Génération du dataset")
    print(f"  Source  : {raw_data_dir}/")
    print(f"  Dest    : {dataset_dir}/")
    print(f"  Séquence: {sequence_length} frames par séquence")
    print(f"{'='*60}\n")

    total_sequences = 0
    stats = {}

    with mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as holistic:

        for sign_folder in sorted(sign_folders):
            sign_name = sign_folder.name
            out_dir   = dataset_path / sign_name
            out_dir.mkdir(parents=True, exist_ok=True)

            print(f"▶ Signe : {sign_name}")

            # Compter le prochain index de séquence disponible (permet de relancer sans écraser)
            existing = list(out_dir.glob("*.npy"))
            seq_index = len(existing)

            sign_count = 0

            # Parcourir tous les fichiers du dossier
            files = sorted(sign_folder.iterdir())
            if not files:
                print(f"  [ATTENTION] Dossier vide : {sign_folder}")
                continue

            for file_path in files:
                suffix = file_path.suffix.lower()

                if suffix in VIDEO_EXTENSIONS:
                    # ── Vidéo ──────────────────────────────────────────────
                    sequences = video_to_sequences(
                        str(file_path), sequence_length, holistic, verbose
                    )
                    for seq in sequences:
                        npy_path = out_dir / f"{seq_index}.npy"
                        np.save(str(npy_path), seq)
                        seq_index  += 1
                        sign_count += 1

                elif suffix in IMAGE_EXTENSIONS:
                    # ── Image ──────────────────────────────────────────────
                    seq = image_to_sequence(
                        str(file_path), sequence_length, holistic, verbose
                    )
                    if seq is not None:
                        npy_path = out_dir / f"{seq_index}.npy"
                        np.save(str(npy_path), seq)
                        seq_index  += 1
                        sign_count += 1

                else:
                    if verbose:
                        print(f"  [IGNORÉ] {file_path.name} (extension non supportée)")

            stats[sign_name] = sign_count
            total_sequences += sign_count
            print(f"  ✓ {sign_count} séquence(s) générée(s) → {out_dir}/\n")

    # ── Résumé ────────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Résumé du dataset généré")
    print(f"{'='*60}")
    for sign, count in stats.items():
        bar = "█" * count
        print(f"  {sign:<15} {bar} ({count})")
    print(f"\n  Total : {total_sequences} séquences dans {dataset_dir}/")

    # Sauvegarder aussi la liste des labels pour l'entraînement
    labels = sorted(stats.keys())
    labels_path = dataset_path / "labels.npy"
    np.save(str(labels_path), np.array(labels))
    print(f"  Labels sauvegardés → {labels_path}")
    print(f"{'='*60}\n")


# ─── Point d'entrée CLI ───────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EchoSign — Conversion vidéo/image → .npy")
    parser.add_argument(
        "--raw_data",
        default="raw_data",
        help="Dossier source contenant les sous-dossiers par signe (défaut: raw_data)",
    )
    parser.add_argument(
        "--dataset",
        default="dataset",
        help="Dossier de sortie pour les .npy (défaut: dataset)",
    )
    parser.add_argument(
        "--sequence_length",
        type=int,
        default=DEFAULT_SEQUENCE_LENGTH,
        help=f"Nombre de frames par séquence (défaut: {DEFAULT_SEQUENCE_LENGTH})",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Réduit les messages de log",
    )

    args = parser.parse_args()

    process_all(
        raw_data_dir    = args.raw_data,
        dataset_dir     = args.dataset,
        sequence_length = args.sequence_length,
        verbose         = not args.quiet,
    )