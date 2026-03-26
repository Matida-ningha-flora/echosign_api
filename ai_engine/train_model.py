"""
train_model.py
==============
Entraîne un réseau BiLSTM sur les séquences .npy générées par data_processor.py.

Architecture du modèle :
  Input  → (sequence_length, 1662)
  BiLSTM(128) → BiLSTM(64) → Dense(64) → Dropout → Dense(nb_signes, softmax)

Usage :
  python ai_engine/train_model.py
  python ai_engine/train_model.py --epochs 100 --batch_size 16
  python ai_engine/train_model.py --dataset custom_dataset/ --no_plot
"""

import os
import sys
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau,
    TensorBoard,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

# ─── Constantes ───────────────────────────────────────────────────────────────

SEQUENCE_LENGTH  = 30
KEYPOINTS_DIM    = 1692
DEFAULT_EPOCHS   = 200
DEFAULT_BATCH    = 32
VALIDATION_SPLIT = 0.15
TEST_SPLIT       = 0.10
MIN_SEQUENCES    = 5     # Minimum de séquences par signe pour entraîner correctement

DATASET_DIR  = Path(__file__).parent.parent / "dataset"
MODELS_DIR   = Path(__file__).parent / "models"
LOGS_DIR     = Path(__file__).parent / "logs"


# ─── Chargement du dataset ────────────────────────────────────────────────────

def load_dataset(dataset_dir: Path, sequence_length: int) -> tuple:
    """
    Charge toutes les séquences .npy depuis dataset/<SIGNE>/<n>.npy

    Retourne :
        X      — np.ndarray (N, sequence_length, 1662)
        y      — np.ndarray (N,) labels entiers encodés
        labels — list des noms de signes dans l'ordre
    """
    sign_dirs = sorted([d for d in dataset_dir.iterdir() if d.is_dir()])

    if not sign_dirs:
        raise FileNotFoundError(
            f"Aucun signe trouvé dans {dataset_dir}/\n"
            f"Lance d'abord : python ai_engine/data_processor.py"
        )

    all_sequences = []
    all_labels    = []
    label_names   = []
    skipped       = []

    print(f"\n{'='*60}")
    print(f"  Chargement du dataset depuis {dataset_dir}/")
    print(f"{'='*60}")

    for sign_dir in sign_dirs:
        npy_files = sorted(sign_dir.glob("*.npy"))
        if not npy_files:
            skipped.append(sign_dir.name)
            continue

        sign_seqs = []
        for npy_file in npy_files:
            seq = np.load(str(npy_file))

            # Vérifier / adapter la forme
            if seq.shape == (sequence_length, KEYPOINTS_DIM):
                sign_seqs.append(seq)
            elif seq.ndim == 2 and seq.shape[1] == KEYPOINTS_DIM:
                # Séquence de longueur différente → resize
                if seq.shape[0] >= sequence_length:
                    sign_seqs.append(seq[:sequence_length])
                else:
                    # Pad avec la dernière frame
                    pad = np.tile(seq[-1], (sequence_length - seq.shape[0], 1))
                    sign_seqs.append(np.vstack([seq, pad]))
            else:
                print(f"  [IGNORÉ] {npy_file.name} — forme inattendue {seq.shape}")
                continue

        count = len(sign_seqs)
        bar   = "█" * min(count, 40)
        print(f"  {sign_dir.name:<15} {bar} {count} séquences")

        if count < MIN_SEQUENCES:
            print(f"             ↳ [ATTENTION] Seulement {count} séquences — recommandé : {MIN_SEQUENCES}+")

        label_idx = len(label_names)
        label_names.append(sign_dir.name)
        all_sequences.extend(sign_seqs)
        all_labels.extend([label_idx] * count)

    if skipped:
        print(f"\n  [IGNORÉ] Dossiers vides : {', '.join(skipped)}")

    if not all_sequences:
        raise ValueError("Aucune séquence valide trouvée dans le dataset.")

    X = np.array(all_sequences, dtype=np.float32)  # (N, seq_len, 1662)
    y = np.array(all_labels,    dtype=np.int32)     # (N,)

    print(f"\n  Total : {len(X)} séquences | {len(label_names)} signes")
    print(f"  Shape X : {X.shape}")
    print(f"{'='*60}\n")

    return X, y, label_names


# ─── Construction du modèle BiLSTM ───────────────────────────────────────────

def build_model(sequence_length: int, n_keypoints: int, n_classes: int) -> keras.Model:
    """
    BiLSTM empilé avec attention légère.
    Robuste pour les séquences de signes.
    """
    inputs = keras.Input(shape=(sequence_length, n_keypoints), name="keypoints")

    # Normalisation des features (stabilise l'entraînement)
    x = layers.LayerNormalization()(inputs)

    # Couche 1 — BiLSTM large (capture les patterns temporels longs)
    x = layers.Bidirectional(
        layers.LSTM(128, return_sequences=True, dropout=0.2, recurrent_dropout=0.2),
        name="bilstm_1"
    )(x)

    # Couche 2 — BiLSTM plus fine
    x = layers.Bidirectional(
        layers.LSTM(64, return_sequences=True, dropout=0.2, recurrent_dropout=0.2),
        name="bilstm_2"
    )(x)

    # Couche 3 — BiLSTM de sortie
    x = layers.Bidirectional(
        layers.LSTM(32, return_sequences=False, dropout=0.2),
        name="bilstm_3"
    )(x)

    # Fully connected
    x = layers.Dense(64, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Dense(32, activation="relu")(x)
    x = layers.Dropout(0.2)(x)

    # Sortie
    outputs = layers.Dense(n_classes, activation="softmax", name="predictions")(x)

    model = keras.Model(inputs, outputs, name="EchoSign_BiLSTM")
    return model


# ─── Callbacks ────────────────────────────────────────────────────────────────

def get_callbacks(models_dir: Path, logs_dir: Path) -> list:
    models_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    return [
        # Sauvegarde automatique du meilleur modèle
        ModelCheckpoint(
            filepath          = str(models_dir / "action_best.h5"),
            monitor           = "val_accuracy",
            mode              = "max",
            save_best_only    = True,
            verbose           = 1,
        ),
        # Arrêt si plus d'amélioration après N epochs
        EarlyStopping(
            monitor   = "val_accuracy",
            patience  = 30,
            mode      = "max",
            verbose   = 1,
            restore_best_weights = True,
        ),
        # Réduction du learning rate si plateau
        ReduceLROnPlateau(
            monitor  = "val_loss",
            factor   = 0.5,
            patience = 10,
            min_lr   = 1e-6,
            verbose  = 1,
        ),
        # TensorBoard (optionnel mais utile)
        TensorBoard(
            log_dir           = str(logs_dir / timestamp),
            histogram_freq    = 0,
            write_graph       = False,
        ),
    ]


# ─── Évaluation ───────────────────────────────────────────────────────────────

def evaluate_model(model, X_test, y_test, label_names):
    """Affiche le rapport de classification complet sur le jeu de test."""
    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)

    print(f"\n{'='*60}")
    print("  Rapport de classification (jeu de test)")
    print(f"{'='*60}")
    print(classification_report(y_test, y_pred, target_names=label_names))

    # Accuracy globale
    acc = np.mean(y_pred == y_test)
    print(f"  Accuracy test : {acc*100:.2f}%")

    # Matrice de confusion simplifiée
    cm = confusion_matrix(y_test, y_pred)
    print(f"\n  Matrice de confusion :")
    header = "  " + "".join(f"{name[:5]:>7}" for name in label_names)
    print(header)
    for i, row in enumerate(cm):
        row_str = "  " + f"{label_names[i][:5]:>5} " + "".join(f"{v:>7}" for v in row)
        print(row_str)
    print(f"{'='*60}\n")

    return acc


# ─── Pipeline principal ───────────────────────────────────────────────────────

def train(
    dataset_dir:     str = str(DATASET_DIR),
    models_dir:      str = str(MODELS_DIR),
    sequence_length: int = SEQUENCE_LENGTH,
    epochs:          int = DEFAULT_EPOCHS,
    batch_size:      int = DEFAULT_BATCH,
    show_plot:       bool = True,
):
    dataset_path = Path(dataset_dir)
    models_path  = Path(models_dir)

    # ── 1. Charger le dataset ──────────────────────────────────────────────────
    X, y, label_names = load_dataset(dataset_path, sequence_length)
    n_classes = len(label_names)

    # ── 2. Encoder les labels en one-hot ──────────────────────────────────────
    y_cat = keras.utils.to_categorical(y, num_classes=n_classes)

    # ── 3. Split train / val / test ───────────────────────────────────────────
    X_temp, X_test, y_temp, y_test_cat = train_test_split(
        X, y_cat, test_size=TEST_SPLIT, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=VALIDATION_SPLIT / (1 - TEST_SPLIT),
        random_state=42,
        stratify=np.argmax(y_temp, axis=1),
    )

    print(f"  Train : {len(X_train)} | Val : {len(X_val)} | Test : {len(X_test)}")

    # ── 4. Construire le modèle ────────────────────────────────────────────────
    model = build_model(sequence_length, KEYPOINTS_DIM, n_classes)
    model.summary()

    model.compile(
        optimizer = keras.optimizers.Adam(learning_rate=1e-3),
        loss      = "categorical_crossentropy",
        metrics   = ["accuracy"],
    )

    # ── 5. Entraîner ──────────────────────────────────────────────────────────
    print(f"\n[Train] Démarrage — {epochs} epochs max, batch {batch_size}")
    print(f"[Train] Signes : {label_names}\n")

    history = model.fit(
        X_train, y_train,
        validation_data = (X_val, y_val),
        epochs          = epochs,
        batch_size      = batch_size,
        callbacks       = get_callbacks(models_path, LOGS_DIR),
        verbose         = 1,
    )

    # ── 6. Évaluer sur le jeu de test ─────────────────────────────────────────
    y_test_int = np.argmax(y_test_cat, axis=1)
    test_acc = evaluate_model(model, X_test, y_test_int, label_names)

    # ── 7. Sauvegarder le modèle final ────────────────────────────────────────
    models_path.mkdir(parents=True, exist_ok=True)

    final_path  = models_path / "action.h5"
    labels_path = models_path / "labels.npy"

    model.save(str(final_path))
    np.save(str(labels_path), np.array(label_names))

    print(f"\n[Train] ✓ Modèle sauvegardé → {final_path}")
    print(f"[Train] ✓ Labels sauvegardés → {labels_path}")
    print(f"[Train] ✓ Accuracy test finale : {test_acc*100:.2f}%")
    print(f"\n[Train] Lance le serveur avec :")
    print(f"        uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload\n")

    # ── 8. Courbes d'entraînement (optionnel) ─────────────────────────────────
    if show_plot:
        try:
            import matplotlib.pyplot as plt

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
            fig.suptitle("EchoSign — Entraînement BiLSTM", fontsize=14)

            ax1.plot(history.history["accuracy"],     label="Train")
            ax1.plot(history.history["val_accuracy"], label="Validation")
            ax1.set_title("Accuracy")
            ax1.set_xlabel("Epoch")
            ax1.set_ylabel("Accuracy")
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            ax2.plot(history.history["loss"],     label="Train")
            ax2.plot(history.history["val_loss"], label="Validation")
            ax2.set_title("Loss")
            ax2.set_xlabel("Epoch")
            ax2.set_ylabel("Loss")
            ax2.legend()
            ax2.grid(True, alpha=0.3)

            plot_path = models_path / "training_curves.png"
            plt.tight_layout()
            plt.savefig(str(plot_path), dpi=120)
            print(f"[Train] ✓ Courbes sauvegardées → {plot_path}")
            plt.show()

        except ImportError:
            print("[Train] matplotlib non installé — courbes non générées.")
            print("        pip install matplotlib  pour les activer")

    return model, history, label_names


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EchoSign — Entraînement BiLSTM")
    parser.add_argument("--dataset",         default=str(DATASET_DIR))
    parser.add_argument("--models_dir",      default=str(MODELS_DIR))
    parser.add_argument("--sequence_length", type=int, default=SEQUENCE_LENGTH)
    parser.add_argument("--epochs",          type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch_size",      type=int, default=DEFAULT_BATCH)
    parser.add_argument("--no_plot",         action="store_true")

    args = parser.parse_args()

    train(
        dataset_dir     = args.dataset,
        models_dir      = args.models_dir,
        sequence_length = args.sequence_length,
        epochs          = args.epochs,
        batch_size      = args.batch_size,
        show_plot       = not args.no_plot,
    )
