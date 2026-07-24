# 🧠 Prédiction de Volatilité Implicite : Machine Learning vs Deep Learning

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://iv-prediction-rf-xgb-mlp.streamlit.app/)

## 📌 Projet
La modélisation de la volatilité implicite ($\sigma$) est un enjeu central en finance quantitative et en gestion des risques. Ce projet compare et évalue les capacités d'apprentissage d'algorithmes avancés face aux imperfections du modèle analytique de Black-Scholes (notamment le *Smile de volatilité* et la *structure à terme*).

Cette application interactive permet de comparer en temps réel les prédictions de trois modèles (**Random Forest**, **XGBoost** et un **Réseau de Neurones PyTorch MLP**) entraînés sur un jeu de données simulé de 100 000 options européennes, tout en intégrant une couche d'explicabilité financière (XAI).

---

## ✨ Fonctionnalités clés
* **Benchmark Multi-Modèles :** Comparaison instantanée de la précision de tarification. Le réseau de neurones PyTorch surpasse les méthodes arborescentes avec un score $R^2 > 97\%$ et une erreur moyenne absolue (MAE) de $1.4\%$.
* **Animation 4D ("Respiration du Marché") :** Visualisation interactive et dynamique de la déformation de la surface de volatilité complète lors d'un choc sur le prix du sous-jacent $S_0$ (analyse des régimes *Sticky-Strike* vs *Sticky-Delta*).
* **Surfaces 3D vectorisées des Grecques :** Génération spatio-temporelle interactive des sensibilités de l'option ($\Delta$, $\Gamma$, $\nu$, $\theta$, $\rho$) calculées dynamiquement à partir des prédictions continues du modèle Deep Learning.
* **Cartographie des Régimes de Domination :** Graphique 3D isolant les zones de l'espace (Strike / Maturité) où les arbres de décision performent mieux que le réseau de neurones (et inversement).
* **Explicabilité du Modèle (XAI) :** Intégration de l'algorithme **SHAP** (*TreeExplainer* et *DeepExplainer*) pour décrypter le poids des variables (Moneyness, Maturité, Taux) et justifier localement chaque prédiction (graphiques Waterfall).

---

## 🛠️ Stack technique
* **Deep Learning & ML :** PyTorch, XGBoost, Scikit-Learn
* **Explainable AI (XAI) :** SHAP, Joblib
* **Data Visualisation :** Plotly (3D & Express), Matplotlib, Seaborn
* **Interface & Cloud :** Streamlit Community Cloud

---

## 📊 Performances des modèles (sur Test Set)

| Modèle | Score $R^2$ | MAE (Erreur Moyenne) | Avantage principal observé |
| :--- | :--- | :--- | :--- |
| **PyTorch MLP** | **0.9701** | **0.0146** | Lissage continu parfait pour les surfaces géométriques et le Smile. |
| **XGBoost Opt** | 0.9637 | 0.0171 | Robustesse accrue sur les strikes extrêmes et inférence rapide. |
| **Random Forest Opt** | 0.9122 | 0.0303 | Grande stabilité globale (Bagging) et interprétabilité directe. |

---

## 🚀 Installation et exécution locale

1. **Cloner le dépôt :**
   git clone [https://github.com/Leandr75e/Implied-Volatility-Prediction---RF-vs-XGB-vs-MLP.git](https://github.com/Leandr75e/Implied-Volatility-Prediction---RF-vs-XGB-vs-MLP.git)
   cd Implied-Volatility-Prediction---RF-vs-XGB-vs-MLP

2. **Installer les dépendances :**
    pip install -r requirements.txt

3. **Lancer l'application interactive :**
    streamlit run app.py
