# EchoSign API — Guide d'utilisation

## 1. Structure des dossiers à créer manuellement

```
Echosign_api/
├── raw_data/               ← TU CRÉES CE DOSSIER
│   ├── BONJOUR/            ← Nom du signe EN MAJUSCULES
│   │   ├── clip1.mp4
│   │   ├── clip2.avi
│   │   └── photo.jpg
│   ├── MERCI/
│   │   └── merci.mp4
│   └── A/                  ← Pour les lettres de l'alphabet
│       ├── lettre_a.jpg
│       └── lettre_a2.png
│
├── dataset/                ← Généré automatiquement par data_processor.py
├── ai_engine/
│   └── models/             ← Généré automatiquement par train_model.py
└── app/
```

---

## 2. Comment insérer tes vidéos et images

### Règles de nommage :

| Règle | Correct | Incorrect |
|-------|---------|-----------|
| Dossier = nom du signe | `BONJOUR/` | `bonjour/` |
| Fichier = peu importe | `clip1.mp4`, `test.jpg` | — |
| Formats vidéo acceptés | `.mp4` `.avi` `.mov` `.mkv` | `.wmv` `.flv` |
| Formats image acceptés | `.jpg` `.jpeg` `.png` `.bmp` | `.gif` `.tiff` |

### Minimum recommandé par signe :
- **Vidéos** : 5-10 clips de 2-5 secondes, filmés sous des angles différents
- **Images** : 20-30 photos (pour les gestes statiques comme les lettres A-Z)
- **Pour une bonne précision** : 50+ séquences par signe après extraction

---

## 3. Workflow complet (à suivre dans cet ordre)

### Étape 1 — Installer les dépendances
```bash
pip install -r requirements.txt
```

### Étape 2 — Mettre les vidéos/images dans raw_data/
```
raw_data/
├── BONJOUR/
│   └── video1.mp4
└── MERCI/
    └── video1.mp4
```

### Étape 3 — Générer les séquences .npy
```bash
# Depuis la racine Echosign_api/
python ai_engine/data_processor.py

# Options avancées :
python ai_engine/data_processor.py --sequence_length 45   # si tu veux 45 frames
python ai_engine/data_processor.py --quiet                # moins de messages
```

Résultat attendu :
```
dataset/
├── BONJOUR/
│   ├── 0.npy   ← séquence (30, 1662)
│   ├── 1.npy
│   └── 2.npy
├── MERCI/
│   └── 0.npy
└── labels.npy  ← ['BONJOUR', 'MERCI']
```

### Étape 4 — Entraîner le modèle
```bash
python ai_engine/train_model.py
```
Génère : `ai_engine/models/action.h5` + `ai_engine/models/labels.npy`

### Étape 5 — Démarrer le serveur
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### Étape 6 — Tester l'API
```bash
# Santé du serveur
curl http://localhost:8000/health

# Liste des signes reconnus
curl http://localhost:8000/signs
```

---

## 4. Test du WebSocket depuis Python (sans Flutter)

Crée un fichier `test_websocket.py` à la racine :

```python
import asyncio
import websockets
import json
import numpy as np

async def test():
    uri = "ws://localhost:8000/ws/recognize"
    async with websockets.connect(uri) as ws:
        # Envoyer 30 frames de keypoints aléatoires (pour tester la connexion)
        for i in range(35):
            fake_keypoints = np.zeros(1662).tolist()
            await ws.send(json.dumps({"keypoints": fake_keypoints}))
            response = json.loads(await ws.recv())
            print(f"Frame {i+1}: {response}")

asyncio.run(test())
```

```bash
pip install websockets
python test_websocket.py
```

---

## 5. Code Flutter minimal (connexion WebSocket)

```dart
import 'dart:convert';
import 'package:web_socket_channel/web_socket_channel.dart';

class EchoSignService {
  final channel = WebSocketChannel.connect(
    Uri.parse('ws://TON_IP:8000/ws/recognize'),
  );

  // Envoyer une frame de keypoints
  void sendFrame(List<double> keypoints) {
    channel.sink.add(jsonEncode({"keypoints": keypoints}));
  }

  // Réinitialiser le buffer (entre deux signes)
  void reset() {
    channel.sink.add(jsonEncode({"reset": true}));
  }

  // Écouter les résultats
  Stream get results => channel.stream.map((msg) => jsonDecode(msg));

  void dispose() => channel.sink.close();
}
```

---

## 6. Adresse IP pour Flutter

| Contexte | URL à utiliser |
|----------|----------------|
| Émulateur Android | `ws://10.0.2.2:8000/ws/recognize` |
| iPhone Simulator | `ws://localhost:8000/ws/recognize` |
| Appareil physique (même WiFi) | `ws://192.168.X.X:8000/ws/recognize` |
| Production | `wss://ton-domaine.com/ws/recognize` |

Pour connaître ton IP locale : `ipconfig` (Windows) ou `ifconfig` (Linux/Mac)