import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import time
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV

def black_scholes_call(S0, K, T, r, sigma):

    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    call_price=S0*norm.cdf(d1)-K*np.exp(-r*T)*norm.cdf(d2)

    return call_price

N_samples = 100000
np.random.seed(42)

S0_simulated = 100 #invariant par échelle
K_samples = np.random.uniform(70.0, 130.0, size = N_samples)
T_samples = np.random.uniform(0.1, 3.0, size = N_samples)
sigma_samples = np.random.uniform(0.05, 0.60, size = N_samples)
r_samples = np.random.uniform(0.0, 0.08, size = N_samples)

price_samples = black_scholes_call(S0_simulated, K_samples, T_samples, r_samples, sigma_samples)

noise = np.random.normal(0, 0.02, size = N_samples)
price_samples_noisy = price_samples * (1+noise)
price_samples_noisy = np.clip(price_samples_noisy, 0, None)

df = pd.DataFrame({
    'log-moneyness': np.log(S0_simulated / K_samples),
    'Strike Price (K)': K_samples,
    'Maturity (T)': T_samples,
    'Risk-free Rate (r)': r_samples,
    'Call Option Price': price_samples_noisy,
    'Volatility (sigma)': sigma_samples
})

print("Aperçu des données simulées (S0 = 100) :")
print(df.head())
print("-" * 100)
print("\nStatistiques descriptives :")
print(df.describe())

X = df[['log-moneyness', 'Strike Price (K)', 'Maturity (T)', 'Risk-free Rate (r)', 'Call Option Price']]
y = df['Volatility (sigma)']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42)

print('-' * 100)
print('Split des données en ensembles d\'entraînement et de test :')
print(f"Taille de la base d'entraînement : {X_train.shape[0]} lignes")
print(f"Taille de la base de test : {X_test.shape[0]} lignes")
print('-' * 100)

print("Entraînement du modèle de régression Random Forest...")
start_time = time.time()
rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)

rf_model.fit(X_train, y_train)

end_time = time.time()
print(f"Temps d'entrainement du modèle : {end_time - start_time: .2f} secondes")

y_pred = rf_model.predict(X_test)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print("\nEntraînement du modèle XGBoost en cours...")
start_xgb = time.time()

model_xgb = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42, n_jobs=-1)
model_xgb.fit(X_train, y_train)

end_xgb = time.time()
print(f"Temps d'entrainement du modèle XGBoost : {end_xgb - start_xgb: .2f} secondes")

y_pred_xgb = model_xgb.predict(X_test)
r2_xgb = r2_score(y_test, y_pred_xgb)
mae_xgb = mean_absolute_error(y_test, y_pred_xgb)

np.random.seed(42)
idx = np.random.choice(len(y_test), size=1000, replace=False)


# Recherche d'hyperparamètres

X_train_sample = X_train.sample(n=10000, random_state=42)
y_train_sample = y_train.loc[X_train_sample.index]

print("\n" + "="*100)
print("1. RECHERCHE DES HYPERPARAMÈTRES POUR RANDOM FOREST")
print("="*100)

param_grid_rf = {
    'n_estimators': [40, 60, 80],
    'max_depth': [3, 5, 8, 13],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'bootstrap': [True, False],
}

grid_rf = GridSearchCV(
    estimator=RandomForestRegressor(random_state=42),
    param_grid=param_grid_rf, 
    cv=5,scoring='neg_mean_absolute_error', 
    n_jobs=-1, 
    verbose=2)

start_time = time.time()

grid_rf.fit(X_train_sample, y_train_sample)

best_rf_model = grid_rf.best_estimator_
best_rf_model.fit(X_train, y_train)

y_pred_rf = best_rf_model.predict(X_test)

print("\n" + "="*100)
print("2. RECHERCHE DES HYPERPARAMÈTRES POUR XGBOOST")
print("="*100)

param_grid_xgb = {
'n_estimators': [50, 100, 200],
'max_depth': [3, 5, 8, None],
'min_child_weight': [1, 2],
'subsample': [0.8, 1.0],
'learning_rate': [0.01, 0.1, 0.2]
}

grid_xgb = GridSearchCV(
    estimator=XGBRegressor(random_state=42), 
    param_grid = param_grid_xgb, 
    cv=5, scoring = 'neg_mean_absolute_error', 
    n_jobs=-1, 
    verbose = 2)

start_xgb = time.time()

grid_xgb.fit(X_train_sample, y_train_sample)

best_xgb_model = grid_xgb.best_estimator_
best_xgb_model.fit(X_train, y_train)

y_pred_xgb_opt = best_xgb_model.predict(X_test)

r2_rf_opt = r2_score(y_test, y_pred_rf)
r2_xgb_opt = r2_score(y_test, y_pred_xgb_opt)
mae_xgb_opt = mean_absolute_error(y_test, y_pred_xgb_opt)
mae_rf_opt = mean_absolute_error(y_test, y_pred_rf)

print("\n--- Résultats de l'Évaluation pour Random Forest (Base) ---")
print(f"Score R² : {r2:.5f} (Proche de 1 = Excellent)")
print(f"Erreur Moyenne (MAE) : {mae:.5f} (soit {mae*100:.2f}% d'erreur moyenne sur la volatilité)")

print("\n")
print('-' * 100)

print(f"Temps de recherche RF : {time.time() - start_time: .2f} secondes")
print(f"Meilleurs paramètres RF trouvés : {grid_rf.best_params_}")

print("\n" + "-"*100)

print(f"\n--- Évaluation du Random Forest Optimisé ---")
print(f"Score R² : {r2_score(y_test, y_pred_rf):.5f}")
print(f"MAE : {mean_absolute_error(y_test, y_pred_rf):.5f}")

print("\n")
print("="*100)


print("\n--- Résultats de l'Évaluation pour XGBoost (Base) ---")
print(f"Score R² : {r2_xgb:.5f} (Proche de 1 = Excellent)")
print(f"Erreur Moyenne (MAE) : {mae_xgb:.5f} (soit {mae_xgb*100:.2f}% d'erreur moyenne sur la volatilité)")

print("\n" + "-"*100)

print(f"Temps de recherche XGB : {time.time() - start_xgb: .2f} secondes")
print(f"Meilleurs paramètres XGB trouvés : {grid_xgb.best_params_}")

print("\n" + "-"*100)

print(f"\n--- Évaluation du XGBoost Optimisé ---")
print(f"Score R² : {r2_xgb_opt:.5f}")
print(f"MAE : {mean_absolute_error(y_test, y_pred_xgb_opt):.5f}")

plt.figure(figsize=(15, 10))

plt.subplot(2, 2, 1)
plt.scatter(y_test.iloc[idx], y_pred[idx], alpha=0.4, color='blue', label='Prédictions RF Base')
plt.plot([0.05, 0.60], [0.05, 0.60], color='red', linestyle='--', lw=2, label='Idéal (y = x)')
plt.title(f'Random Forest (Base) - R²: {r2:.4f}')
plt.xlabel('Vraie Volatilité')
plt.ylabel('Volatilité Prédite')
plt.grid(True, alpha=0.3)
plt.legend()

# 2. XGBoost Base
plt.subplot(2, 2, 2)
plt.scatter(y_test.iloc[idx], y_pred_xgb[idx], alpha=0.4, color='green', label='Prédictions XGB Base')
plt.plot([0.05, 0.60], [0.05, 0.60], color='red', linestyle='--', lw=2, label='Idéal (y = x)')
plt.title(f'XGBoost (Base) - R²: {r2_xgb:.4f}')
plt.xlabel('Vraie Volatilité')
plt.ylabel('Volatilité Prédite')
plt.grid(True, alpha=0.3)
plt.legend()

# 3. Random Forest Optimisé
plt.subplot(2, 2, 3)
plt.scatter(y_test.iloc[idx], y_pred_rf[idx], alpha=0.4, color='darkblue', label='Prédictions RF Opt')
plt.plot([0.05, 0.60], [0.05, 0.60], color='red', linestyle='--', lw=2, label='Idéal (y = x)')
plt.title(f'Random Forest (Optimisé) - R²: {r2_rf_opt:.4f}')
plt.xlabel('Vraie Volatilité')
plt.ylabel('Volatilité Prédite')
plt.grid(True, alpha=0.3)
plt.legend()

# 4. XGBoost Optimisé
plt.subplot(2, 2, 4)
plt.scatter(y_test.iloc[idx], y_pred_xgb_opt[idx], alpha=0.4, color='darkgreen', label='Prédictions XGB Opt')
plt.plot([0.05, 0.60], [0.05, 0.60], color='red', linestyle='--', lw=2, label='Idéal (y = x)')
plt.title(f'XGBoost (Optimisé) - R²: {r2_xgb_opt:.4f}')
plt.xlabel('Vraie Volatilité')
plt.ylabel('Volatilité Prédite')
plt.grid(True, alpha=0.3)
plt.legend()

plt.tight_layout()
plt.show()

joblib.dump(best_rf_model, 'rf_volatility_model.pkl')
joblib.dump(best_xgb_model, 'xgb_volatility_model.pkl')


#COMPARAISON MLP


print("\n")
print("="*100)
print("="*100)
print("Réseau de Neuronnes (MLP)")
print("="*100)
print("="*100)
print("\n")

from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import seaborn as sns

def black_scholes_call(S0, K, T, r, sigma):
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return S0*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)

N_samples = 100000
np.random.seed(42)

S0_simulated = 100.0
K_samples = np.random.uniform(70.0, 130.0, size=N_samples)
T_samples = np.random.uniform(0.1, 3.0, size=N_samples)
sigma_samples = np.random.uniform(0.05, 0.60, size=N_samples)
r_samples = np.random.uniform(0.0, 0.08, size=N_samples)

price_samples = black_scholes_call(S0_simulated, K_samples, T_samples, r_samples, sigma_samples)
noise = np.random.normal(0, 0.02, size=N_samples)
price_samples_noisy = np.clip(price_samples * (1 + noise), 0, None)

df = pd.DataFrame({
    'log-moneyness': np.log(S0_simulated / K_samples),
    'Strike Price (K)': K_samples,
    'Maturity (T)': T_samples,
    'Risk-free Rate (r)': r_samples,
    'Call Option Price': price_samples_noisy,
    'Volatility (sigma)': sigma_samples
})

X = df[['log-moneyness', 'Strike Price (K)', 'Maturity (T)', 'Risk-free Rate (r)', 'Call Option Price']]
y = df['Volatility (sigma)']

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.20, random_state=42
)

# 2. On sépare les 20% Temp en deux parts égales (10% Val, 10% Test)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42
)

print("\n"f"Taille Train : {len(X_train)} échantillons")
print(f"Taille Val   : {len(X_val)} échantillons")
print(f"Taille Test  : {len(X_test)} échantillons")

scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1))

X_val_scaled = scaler_X.transform(X_val)
y_val_scaled = scaler_y.transform(y_val.values.reshape(-1, 1))

X_test_scaled = scaler_X.transform(X_test)

train_dataset = TensorDataset(torch.tensor(X_train_scaled, dtype=torch.float32), 
                              torch.tensor(y_train_scaled, dtype=torch.float32))
val_dataset = TensorDataset(torch.tensor(X_val_scaled, dtype=torch.float32), 
                            torch.tensor(y_val_scaled, dtype=torch.float32))

train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False)

class VolatilityMLP(nn.Module):
    def __init__(self, input_size):
        super(VolatilityMLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.GELU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 128),
            nn.GELU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.network(x)

model = VolatilityMLP(input_size=5)
criterion = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)

train_losses = []
val_losses = []
epochs = 100

print("\n""Entraînement du MLP en cours...")
start_time = time.time()

for epoch in range(epochs):
    model.train()
    running_train_loss = 0.0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        prediction = model(batch_X)
        loss = criterion(prediction, batch_y)
        loss.backward()
        optimizer.step()
        running_train_loss += loss.item() * batch_X.size(0)
    
    epoch_train_loss = running_train_loss / len(train_loader.dataset)
    train_losses.append(epoch_train_loss)

    model.eval()
    running_val_loss = 0.0
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            pred_val = model(batch_X)
            v_loss = criterion(pred_val, batch_y)
            running_val_loss += v_loss.item() * batch_X.size(0)
            
    epoch_val_loss = running_val_loss / len(val_loader.dataset)
    val_losses.append(epoch_val_loss)

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1:03d}/{epochs} | Train MSE: {epoch_train_loss:.5f} | Val MSE: {epoch_val_loss:.5f}")

print(f"\nTemps d'entraînement du modèle MLP : {time.time() - start_time:.2f} secondes")

model.eval()
with torch.no_grad():
    y_pred_scaled = model(torch.tensor(X_test_scaled, dtype=torch.float32)).numpy()
    y_pred = scaler_y.inverse_transform(y_pred_scaled).flatten()

r2_mlp = r2_score(y_test, y_pred)
mae_mlp = mean_absolute_error(y_test, y_pred)

print("\n" + "="*50)
print("--- Évaluation du MLP sur le Test Set ---")
print(f"Score R² : {r2_mlp:.5f}")
print(f"MAE : {mae_mlp:.5f}")
print("="*50)

torch.save(model.state_dict(), 'mlp_volatility_model.pth')
joblib.dump(scaler_X, 'scaler_X.pkl')
joblib.dump(scaler_y, 'scaler_y.pkl')
print("\n Modèle MLP et scalers sauvegardés avec succès !")

abs_errors = np.abs(y_test.values - y_pred)

test_df = X_test.copy()
test_df['abs_error'] = abs_errors

plt.figure(figsize=(16, 6))

plt.subplot(1, 2, 1)
plt.plot(range(1, epochs + 1), train_losses, label='Train MSE Loss', color='#1f77b4', lw=2)
plt.plot(range(1, epochs + 1), val_losses, label='Val MSE Loss', color='#ff7f0e', linestyle='--', lw=2)
plt.title("Courbe d'Apprentissage du MLP (Loss Curve)", fontsize=13, fontweight='bold')
plt.xlabel("Epochs")
plt.ylabel("Loss (MSE)")
plt.yscale('log') 
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.legend(fontsize=11)

plt.subplot(1, 2, 2)
test_df['moneyness_bin'] = pd.qcut(test_df['log-moneyness'], q=8, duplicates='drop')
test_df['maturity_bin'] = pd.qcut(test_df['Maturity (T)'], q=8, duplicates='drop')

heatmap_data = test_df.pivot_table(
    index='maturity_bin', 
    columns='moneyness_bin', 
    values='abs_error', 
    aggfunc='mean',
    observed=False 
)

heatmap_data.index = [f"{i.left:.2f} à {i.right:.2f}" for i in heatmap_data.index]
heatmap_data.columns = [f"{c.left:.2f} à {c.right:.2f}" for c in heatmap_data.columns]

sns.heatmap(heatmap_data, cmap='YlOrRd', annot=True, fmt=".4f", cbar_kws={'label': 'Erreur Moyenne Absolue (MAE)'})
plt.title("Carte des Erreurs (Residual Heatmap)\nErreur en fonction de la Maturité et du Log-Moneyness", fontsize=13, fontweight='bold')
plt.xlabel("Log-Moneyness ln(S0/K)")
plt.ylabel("Maturité T (Années)")

plt.tight_layout()
plt.show()

results_df = pd.DataFrame({
    'Modèle': ['Random Forest Opt', 'XGBoost Opt', 'MLP PyTorch'],
    'R² Score': [r2_rf_opt, r2_xgb_opt, r2_mlp],
    'MAE': [mae_rf_opt, mae_xgb_opt, mae_mlp]
})

print("\n" + "="*50)
print("COMPARAISON FINALE SUR LE MÊME JEU DE TEST")
print("="*50)
print(results_df.to_string(index=False))