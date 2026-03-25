"""
main.py
=======
Serveur FastAPI — Point d'entrée du backend EchoSign.

Endpoints disponibles :
  WS  /ws/recognize         — Temps réel : Flutter envoie frame par frame
  POST /predict/sequence     — REST : Flutter envoie une séquence complète d'un coup
  GET  /health               — Vérification que le serveur tourne
  GET  /signs                — Liste des signes que le modèle connaît

Lancement :
  uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
"""

import sys
import os
import json
import time
import traceback
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Assure que les imports relatifs fonctionnent quelle que soit la façon de lancer
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.processor import clean_frame, clean_sequence, ProcessorError
from app.inference  import get_engine, init_engine, EchoSignInference


# ─── Démarrage / arrêt ────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Charge le modèle au démarrage, libère les ressources à l'arrêt."""
    print("\n[EchoSign] Démarrage du serveur...")
    try:
        engine = get_engine()
        app.state.engine = engine
        print(f"[EchoSign] ✓ Prêt — {engine.sign_count} signes chargés")
    except FileNotFoundError as e:
        print(f"[EchoSign] ✗ ERREUR : {e}")
        print("[EchoSign]   Le serveur démarre quand même — /predict retournera des erreurs")
        app.state.engine = None
    yield
    print("[EchoSign] Arrêt du serveur.")


# ─── Application ──────────────────────────────────────────────────────────────

app = FastAPI(
    title       = "EchoSign Vision API",
    description = "Traduction du Langage des Signes en texte — Backend IA",
    version     = "1.0.0",
    lifespan    = lifespan,
)

# CORS — autorise Flutter (mobile + web) et le dev local
app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],   # En production : remplace par ton domaine
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)


# ─── Utilitaire : récupérer le moteur ─────────────────────────────────────────

def _get_engine() -> EchoSignInference:
    engine = app.state.engine
    if engine is None or not engine.is_loaded:
        raise HTTPException(
            status_code = 503,
            detail      = "Modèle non chargé. Lance d'abord python ai_engine/train_model.py",
        )
    return engine


# ─── GET /health ──────────────────────────────────────────────────────────────

@app.get("/health", summary="Vérification du serveur")
async def health():
    """
    Répond toujours 200. Flutter peut l'appeler avant de se connecter
    pour vérifier que le serveur est en ligne.
    """
    engine = app.state.engine
    return {
        "status"      : "ok",
        "model_loaded": engine is not None and engine.is_loaded,
        "sign_count"  : engine.sign_count if engine else 0,
        "timestamp"   : time.time(),
    }


# ─── GET /signs ───────────────────────────────────────────────────────────────

@app.get("/signs", summary="Liste des signes reconnus")
async def get_signs():
    """Retourne la liste des signes que le modèle sait reconnaître."""
    engine = _get_engine()
    return {
        "signs": engine.sign_list,
        "count": engine.sign_count,
    }


# ─── POST /predict/sequence ───────────────────────────────────────────────────

@app.post("/predict/sequence", summary="Prédiction depuis une séquence complète")
async def predict_sequence(payload: dict):
    """
    Reçoit une séquence complète (30 frames × 1662 keypoints) et retourne
    le signe prédit. Utile pour les tests et pour les appareils qui bufferisent
    côté Flutter avant d'envoyer.

    Body JSON attendu :
    {
        "sequence": [[kp1, kp2, ..., kp1662], ...]   // 30 listes de 1662 floats
    }

    Réponse :
    {
        "label": "BONJOUR",
        "confidence": 0.97,
        "top3": [{"label": "BONJOUR", "score": 0.97}, ...],
        "ready": true
    }
    """
    engine = _get_engine()

    try:
        sequence = clean_sequence(payload)
    except ProcessorError as e:
        raise HTTPException(status_code=422, detail=str(e))

    result = engine.predict_sequence(sequence)
    return result


# ─── WebSocket /ws/recognize ──────────────────────────────────────────────────

@app.websocket("/ws/recognize")
async def ws_recognize(websocket: WebSocket):
    """
    Canal WebSocket temps réel.

    Flutter envoie frame par frame (JSON) :
    {
        "keypoints": [kp1, kp2, ..., kp1662],   // 1662 floats
        "reset": false                            // true pour vider le buffer
    }

    Le serveur répond après chaque frame :
    {
        "label": "BONJOUR",        // null si buffer pas encore plein ou confiance faible
        "confidence": 0.97,
        "top3": [...],
        "ready": true,
        "buffer_fill": 30          // frames accumulées dans le buffer
    }

    Erreur :
    {
        "error": "message d'erreur"
    }
    """
    await websocket.accept()
    engine = app.state.engine

    if engine is None or not engine.is_loaded:
        await websocket.send_json({
            "error": "Modèle non disponible. Le serveur n'est pas prêt."
        })
        await websocket.close()
        return

    client_id = websocket.client.host if websocket.client else "unknown"
    print(f"[WS] Connexion établie depuis {client_id}")

    try:
        while True:
            # ── Réception du message Flutter ──────────────────────────────────
            raw_message = await websocket.receive_text()

            try:
                payload = json.loads(raw_message)
            except json.JSONDecodeError:
                await websocket.send_json({"error": "JSON invalide"})
                continue

            # ── Commande reset (vider le buffer entre deux mots) ──────────────
            if payload.get("reset", False):
                engine.reset_buffer()
                await websocket.send_json({
                    "message": "Buffer réinitialisé",
                    "ready": False,
                    "buffer_fill": 0,
                })
                continue

            # ── Valider les keypoints ──────────────────────────────────────────
            try:
                keypoints = clean_frame(payload)
            except ProcessorError as e:
                await websocket.send_json({"error": str(e)})
                continue

            # ── Ajouter au buffer et prédire ──────────────────────────────────
            result = engine.add_frame(keypoints)
            await websocket.send_json(result)

    except WebSocketDisconnect:
        print(f"[WS] Client {client_id} déconnecté")

    except Exception as e:
        print(f"[WS] Erreur inattendue : {e}")
        traceback.print_exc()
        try:
            await websocket.send_json({"error": f"Erreur serveur : {str(e)}"})
        except Exception:
            pass


# ─── Lancement direct (dev uniquement) ────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host    = "0.0.0.0",
        port    = 8000,
        reload  = True,       # Auto-reload en dev
        workers = 1,          # 1 seul worker (le modèle BiLSTM n'est pas thread-safe)
    )