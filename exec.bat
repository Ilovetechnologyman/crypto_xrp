@echo off
echo Création de l'environnement virtuel...
python -m venv monenv

echo Activation de l'environnement virtuel...
call monenv\Scripts\activate

echo Installation des dépendances...
pip install numpy pandas requests tensorflow scikit-learn matplotlib

echo Lancement du programme...
python main.py

echo Désactivation de l'environnement virtuel...
deactivate

echo Fin du script. Appuyez sur une touche pour fermer.
pause
