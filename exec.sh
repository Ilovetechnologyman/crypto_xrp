#!/bin/bash

echo "Création de l'environnement virtuel..."
python3 -m venv monenv

echo "Activation de l'environnement virtuel..."
source monenv/bin/activate

echo "Installation des dépendances..."
pip install numpy pandas requests tensorflow scikit-learn matplotlib

echo "Lancement du programme..."
python3 main.py

echo "Désactivation de l'environnement virtuel..."
deactivate

echo "Fin du script."
