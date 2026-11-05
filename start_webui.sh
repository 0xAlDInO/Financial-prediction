#!/usr/bin/env bash
set -e

# Vérification et installation automatique des dépendances requises si manquantes
python3 -c "import flask, flask_cors, plotly, torch, pandas" 2>/dev/null || {
    echo "⚙️ Installation des dépendances requises dans votre environnement (.venv)..."
    pip install -r requirements.txt
}

echo "🚀 Lancement de l'interface visuelle Kronos (BTC/USD M5 & M15)..."
python3 webui/run_sequential_ui.py
