import numpy as np
import pandas as pd
import requests
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt

# Fonction pour récupérer les données des prix XPR (Proton) depuis l'API de CoinGecko
def get_historical_data():
    url = "https://api.coingecko.com/api/v3/coins/proton/market_chart"
    params = {"vs_currency": "usd", "days": "365"}  # Limite de 365 jours pour l'API publique
    response = requests.get(url, params=params)
    data = response.json()

    # Vérification de l'existence des prix dans la réponse de l'API
    if 'prices' not in data:
        print(f"Erreur dans la réponse de l'API: {data['error']['status']['error_message']}")
        return None
    
    prices = [entry[1] for entry in data["prices"]]  # Extraire les prix de la réponse
    return prices

# Fonction pour préparer les données pour le Deep Learning (normalisation et création des séquences)
def prepare_data(prices, time_step=60):
    data = np.array(prices).reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data)

    X, y = [], []
    for i in range(time_step, len(data_scaled)):
        X.append(data_scaled[i-time_step:i, 0])  # Séquence de 60 jours
        y.append(data_scaled[i, 0])  # Prix suivant

    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))  # (samples, time_steps, features)
    
    return X, y, scaler

# Fonction pour construire le modèle LSTM
def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dense(units=1))  # Prédiction du prix suivant
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Fonction pour prolonger les prédictions sur plusieurs jours et analyser la tendance
def extend_predictions(model, last_data, time_steps, scaler, n_days=80):
    predicted_prices = []
    current_data = last_data

    for _ in range(n_days):
        next_day_prediction = model.predict(current_data)  # Prédiction du prochain jour
        predicted_prices.append(next_day_prediction[0][0])  # Sauvegarder la prédiction

        # Mettre à jour la séquence d'entrée en décalant les valeurs
        current_data = np.append(current_data[:, 1:, :], [[next_day_prediction]], axis=1)

    # Inverser la normalisation des prédictions
    predicted_prices = scaler.inverse_transform(np.array(predicted_prices).reshape(-1, 1))

    return predicted_prices

# Fonction principale pour entraîner le modèle et prédire les prix
def main():
    # Récupérer les données des prix
    prices = get_historical_data()
    if prices is None:
        return

    # Préparer les données pour l'entraînement
    time_step = 60  # Utiliser les 60 derniers jours pour prédire le prix suivant
    X, y, scaler = prepare_data(prices, time_step)

    # Diviser les données en ensemble d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # Construire le modèle LSTM
    model = build_lstm_model((X_train.shape[1], 1))

    # Entraîner le modèle
    model.fit(X_train, y_train, epochs=10, batch_size=32)

    # Prédire sur l'ensemble de test
    predicted_prices = model.predict(X_test)

    # Réduire la normalisation des prédictions
    predicted_prices = scaler.inverse_transform(predicted_prices)
    y_test = scaler.inverse_transform(y_test.reshape(-1, 1))

    # Visualisation des résultats
    plt.figure(figsize=(14, 7))
    plt.plot(y_test, color='blue', label='Prix réel')
    plt.plot(predicted_prices, color='red', label='Prix prédit')
    plt.title('Prédiction du prix de XPR (Proton) avec LSTM')
    plt.xlabel('Temps')
    plt.ylabel('Prix (USD)')
    plt.legend()
    plt.show()

    # Prolonger la prédiction pour les prochains n_days (par exemple, 30 jours)
    last_data = X_test[-1:]  # Dernier échantillon de test
    extended_predictions = extend_predictions(model, last_data, time_step, scaler, n_days=30)

    # Affichage des prédictions prolongées
    plt.figure(figsize=(14, 7))
    plt.plot(np.arange(len(prices)), prices, color='blue', label='Prix historique')
    plt.plot(np.arange(len(prices), len(prices) + len(extended_predictions)), extended_predictions, color='orange', label='Prix prédit (prolongé)')
    plt.title('Prolongation des prédictions des prix de XPR (Proton)')
    plt.xlabel('Temps')
    plt.ylabel('Prix (USD)')
    plt.legend()
    plt.show()

    # Analyser la tendance des prédictions prolongées (hausse ou baisse)
    trend = "stable"
    if extended_predictions[-1] > extended_predictions[0]:
        trend = "hausse"
    elif extended_predictions[-1] < extended_predictions[0]:
        trend = "baisse"

    print(f"Tendance prédites pour les prochains {len(extended_predictions)} jours : {trend}")

# Appeler la fonction principale
if __name__ == "__main__":
    main()
