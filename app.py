import streamlit as st
import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
import os
import plotly.graph_objects as go
import plotly.express as px
from scipy.stats import norm
import plotly.graph_objects as go
import shap
import matplotlib.pyplot as plt


def black_scholes_call(S0, K, T, r, sigma):
    """Calcule le prix du Call selon Black-Scholes."""
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call_price = S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return call_price

def calculate_call_greeks(S, K, T, r, sigma):
    """
    Calcule les Grecques d'un Call européen selon Black-Scholes.
    """
    if T <= 0 or sigma <= 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0

    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    delta = norm.cdf(d1)
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vega = S * norm.pdf(d1) * np.sqrt(T) / 100  
    theta = (- (S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365 
    rho = (K * T * np.exp(-r * T) * norm.cdf(d2)) / 100 

    return delta, gamma, vega, theta, rho

def calculate_greeks_grid_vectorized(S, K_grid, T_grid, r, sigma_grid):
    """
    Calcule les Grecques de manière vectorisée sur des grilles 2D NumPy (pour les graphiques 3D).
    """
    T_safe = np.maximum(T_grid, 1e-5)
    sigma_safe = np.maximum(sigma_grid, 1e-5)
    
    d1 = (np.log(S / K_grid) + (r + 0.5 * sigma_safe**2) * T_safe) / (sigma_safe * np.sqrt(T_safe))
    d2 = d1 - sigma_safe * np.sqrt(T_safe)
    
    delta = norm.cdf(d1)
    gamma = norm.pdf(d1) / (S * sigma_safe * np.sqrt(T_safe))
    vega = S * norm.pdf(d1) * np.sqrt(T_safe) / 100.0  
    theta = (- (S * norm.pdf(d1) * sigma_safe) / (2 * np.sqrt(T_safe)) - r * K_grid * np.exp(-r * T_safe) * norm.cdf(d2)) / 365.0
    rho = (K_grid * T_safe * np.exp(-r * T_safe) * norm.cdf(d2)) / 100.0
    
    return delta, gamma, vega, theta, rho

st.set_page_config(page_title="Prédiction Volatilité Implicite", layout="wide")

st.title("📈 Prédiction de la Volatilité Implicite")
st.markdown("Comparaison interactive des modèles : Random Forest, XGBoost et PyTorch MLP.")

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

@st.cache_resource
def load_artifacts():
    rf_model, xgb_model, mlp_model = None, None, None
    scaler_X, scaler_y = None, None

    if os.path.exists('rf_volatility_model.pkl'):
        rf_model = joblib.load('rf_volatility_model.pkl')
        
    if os.path.exists('xgb_volatility_model.pkl'):
        xgb_model = joblib.load('xgb_volatility_model.pkl')
        
    if os.path.exists('scaler_X.pkl') and os.path.exists('scaler_y.pkl'):
        scaler_X = joblib.load('scaler_X.pkl')
        scaler_y = joblib.load('scaler_y.pkl')
        
    if os.path.exists('mlp_volatility_model.pth'):
        mlp_model = VolatilityMLP(input_size=5)
        mlp_model.load_state_dict(torch.load('mlp_volatility_model.pth', map_location=torch.device('cpu')))
        mlp_model.eval()

    return rf_model, xgb_model, mlp_model, scaler_X, scaler_y

rf_model, xgb_model, mlp_model, scaler_X, scaler_y = load_artifacts()
st.sidebar.header("Paramètres de l'Option")
st.sidebar.info("Ajustez les paramètres pour voir les prédictions s'actualiser en temps réel.")

S0 = 100.0 
st.sidebar.markdown(f"**Prix du Sous-Jacent (S0) :** {S0}")

K = st.sidebar.slider("Strike (K)", min_value=70.0, max_value=130.0, value=100.0, step=1.0)
T = st.sidebar.slider("Maturité (T en années)", min_value=0.1, max_value=3.0, value=1.0, step=0.1)
r = st.sidebar.slider("Taux sans risque (r)", min_value=0.0, max_value=0.08, value=0.04, step=0.005, format="%.3f")
call_price = st.sidebar.slider("Prix du Call observé", min_value=0.1, max_value=50.0, value=10.0, step=0.5)

log_moneyness = np.log(S0 / K)

input_df = pd.DataFrame({
    'log-moneyness': [log_moneyness],
    'Strike Price (K)': [K],
    'Maturity (T)': [T],
    'Risk-free Rate (r)': [r],
    'Call Option Price': [call_price]
})

st.subheader("Vecteur d'Entrée")
st.dataframe(input_df.style.format("{:.4f}"))

banniere_meilleur_modele = st.empty()

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Prédiction Ponctuelle", "Analyse des Grecques", "Comparaison des modèles","Surface de Volatilité", "Explicabilité (XAI)"])
with tab1:
    st.subheader("Prédictions de la Volatilité Implicite (σ)", 
                 help="Volatilité estimée par chaque modèle pour le vecteur d'entrée sélectionné dans la barre latérale.")
    col1, col2, col3 = st.columns(3)
    
    pred_rf = None
    pred_xgb = None
    pred_mlp = None

    with col1:
        st.markdown("### 🌲 Random Forest")
        if rf_model is not None:
            pred_rf = float(rf_model.predict(input_df)[0])
            st.metric(label="Prédiction", value=f"{pred_rf:.2%}")
        else:
            st.warning("Fichier rf_volatility_model.pkl introuvable")

    with col2:
        st.markdown("### 🚀 XGBoost")
        if xgb_model is not None:
            pred_xgb = float(xgb_model.predict(input_df)[0])
            st.metric(label="Prédiction", value=f"{pred_xgb:.2%}")
        else:
            st.warning("Fichier xgb_volatility_model.pkl introuvable")

    with col3:
        st.markdown("### 🧠 PyTorch MLP")
        if mlp_model is not None and scaler_X is not None and scaler_y is not None:
            input_scaled = scaler_X.transform(input_df)
            input_tensor = torch.tensor(input_scaled, dtype=torch.float32)
            
            with torch.no_grad():
                pred_scaled = mlp_model(input_tensor).numpy()
                
            pred_mlp = float(scaler_y.inverse_transform(pred_scaled).flatten()[0])
            st.metric(label="Prédiction", value=f"{pred_mlp:.2%}", 
                      help="Réseau de neurones Perceptron Multi-Couches avec fonctions GELU et BatchNorm.")
        else:
            st.warning("Fichiers MLP ou Scalers introuvables")

    st.divider()

    st.subheader(
        "1. Sensibilité au Strike (Smile de Volatilité)", 
        help="Montre la variation de la volatilité implicite prédite par chaque modèle lorsque le Strike (K) varie, à maturité et taux constants. Permet de comparer le lissage des courbes."
    )
    
    strikes_range = np.linspace(70, 130, 50)
    
    df_smile = pd.DataFrame({
        'log-moneyness': np.log(S0 / strikes_range),
        'Strike Price (K)': strikes_range,
        'Maturity (T)': [T] * 50,
        'Risk-free Rate (r)': [r] * 50,
        'Call Option Price': [call_price] * 50
    })
    
    fig_smile = go.Figure()

    if xgb_model is not None:
        preds_xgb_smile = xgb_model.predict(df_smile)
        fig_smile.add_trace(go.Scatter(x=strikes_range, y=preds_xgb_smile, mode='lines', name='XGBoost', line=dict(color='green')))
        
    if mlp_model is not None:
        smile_scaled = scaler_X.transform(df_smile)
        smile_tensor = torch.tensor(smile_scaled, dtype=torch.float32)
        with torch.no_grad():
            preds_mlp_scaled = mlp_model(smile_tensor).numpy()
        preds_mlp_smile = scaler_y.inverse_transform(preds_mlp_scaled).flatten()
        fig_smile.add_trace(go.Scatter(x=strikes_range, y=preds_mlp_smile, mode='lines', name='PyTorch MLP', line=dict(color='red')))

    fig_smile.update_layout(
        xaxis_title="Strike (K)", 
        yaxis_title="Volatilité Implicite (σ)",
        hovermode="x unified"
    )

    if rf_model is not None:
        preds_rf_smile = rf_model.predict(df_smile)
        fig_smile.add_trace(go.Scatter(
            x=strikes_range, 
            y=preds_rf_smile, 
            mode='lines', 
            name='Random Forest', 
            line=dict(color='blue', dash='dash') 
        ))
    st.plotly_chart(fig_smile, use_container_width=True)
    

    st.divider()

    st.subheader("2. Volatilité implicite prédite par modèle")
    if pred_rf is not None and pred_xgb is not None and pred_mlp is not None:
        df_comp = pd.DataFrame({
            "Modèle": ["Random Forest", "XGBoost", "MLP"],
            "Volatilité Prédite": [pred_rf, pred_xgb, pred_mlp]
        })
        fig = px.bar(df_comp, x="Modèle", y="Volatilité Prédite", color="Modèle", text_auto='.2%')
        fig.update_layout(yaxis_tickformat='.1%')
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Veuillez vous assurer que les 3 modèles sont bien dans le dossier pour afficher le graphique de comparaison.")

    st.divider()

    st.subheader("3. L'Animation 4D : La Respiration du Marché (Choc sur le Sous-Jacent S₀)")
    st.markdown(
        "Ce graphique 4D montre comment la surface de volatilité complète (Strike × Maturité) se déforme en temps réel "
        "lorsque le prix de l'actif sous-jacent ($S_0$) subit une variation brutale (de 80 € à 120 €). "
    )

    if mlp_model is not None and scaler_X is not None and scaler_y is not None:
        K_grid_anim = np.linspace(70, 130, 30)
        T_grid_anim = np.linspace(0.1, 3.0, 30)
        K_mesh_anim, T_mesh_anim = np.meshgrid(K_grid_anim, T_grid_anim)
        K_flat_anim = K_mesh_anim.flatten()
        T_flat_anim = T_mesh_anim.flatten()

        s0_values = np.arange(80, 122, 2) 
        frames = []
        
        z_min, z_max = 1.0, 0.0 
        z_initial = None

        for s0_val in s0_values:
            log_mon_anim = np.log(s0_val / K_flat_anim)
            call_anim = black_scholes_call(s0_val, K_flat_anim, T_flat_anim, r, 0.20)

            df_anim = pd.DataFrame({
                'log-moneyness': log_mon_anim,
                'Strike Price (K)': K_flat_anim,
                'Maturity (T)': T_flat_anim,
                'Risk-free Rate (r)': [r] * len(K_flat_anim),
                'Call Option Price': call_anim
            })

            input_scaled_anim = scaler_X.transform(df_anim)
            with torch.no_grad():
                pred_scaled_anim = mlp_model(torch.tensor(input_scaled_anim, dtype=torch.float32)).numpy()
            vol_matrix_anim = scaler_y.inverse_transform(pred_scaled_anim).flatten().reshape(30, 30)

            if z_initial is None:
                z_initial = vol_matrix_anim

            z_min = min(z_min, vol_matrix_anim.min())
            z_max = max(z_max, vol_matrix_anim.max())

            frames.append(go.Frame(
                data=[go.Surface(z=vol_matrix_anim, x=K_grid_anim, y=T_grid_anim)],
                name=f"S0={s0_val}€"
            ))

        fig_anim = go.Figure(
            data=[go.Surface(
                z=z_initial,
                x=K_grid_anim,
                y=T_grid_anim,
                colorscale='Turbo',
                cmin=z_min,
                cmax=z_max,
                colorbar=dict(title="Vol. Implicite", len=0.75)
            )],
            frames=frames
        )

        fig_anim.update_layout(
            title=dict(
                text="Dynamique du Smile et de la Structure à Terme face au mouvement du sous-jacent", 
                font=dict(size=16), 
                x=0.5, 
                xanchor='center'
            ),
            scene=dict(
                xaxis_title='Strike (K)',
                yaxis_title='Maturité T (Années)',
                zaxis_title='Volatilité (σ)',
                zaxis=dict(range=[z_min * 0.95, z_max * 1.05]), 
                aspectratio=dict(x=1.4, y=1.4, z=0.85),
                camera=dict(eye=dict(x=1.6, y=-1.6, z=1.1))
            ),
            margin=dict(l=10, r=10, b=10, t=50),
            height=700,
            updatemenus=[dict(
                type="buttons",
                showactive=False,
                y=0,
                x=0.05,
                xanchor="right",
                yanchor="top",
                pad=dict(t=45, r=10),
                buttons=[
                    dict(
                        label="▶️ Lecture",
                        method="animate",
                        args=[None, dict(frame=dict(duration=150, redraw=True), transition=dict(duration=50), fromcurrent=True, mode="immediate")]
                    ),
                    dict(
                        label="⏸️ Pause",
                        method="animate",
                        args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate", transition=dict(duration=0))]
                    )
                ]
            )],
            sliders=[dict(
                active=0,
                yanchor="top",
                xanchor="left",
                currentvalue=dict(font=dict(size=15, color="red"), prefix="Prix du sous-jacente : ", visible=True, xanchor="left"),
                transition=dict(duration=50, easing="cubic-in-out"),
                pad=dict(b=10, t=50),
                len=0.85,
                x=0.1,
                y=0,
                steps=[dict(
                    args=[[f"S0={s_val}€"], dict(frame=dict(duration=150, redraw=True), mode="immediate", transition=dict(duration=50))],
                    label=f"{s_val} €",
                    method="animate"
                ) for s_val in s0_values]
            )]
        )

        st.plotly_chart(fig_anim, use_container_width=True)

        st.info("💡 **Analyse (Sticky Strike vs Sticky Delta) :** Lancez l'animation et observez comment le creux de la surface se déplace latéralement le long de l'axe des Strikes au fur et à mesure que le sous jacent monte ou baisse. Ce comportement montre que le modèle MLP a réussi à apprendre une règle d'or des marchés d'options : lorsque le sous-jacent bouge, le smile ne reste pas immobile, il translate pour s'adapter au nouveau niveau de moneyness.")



with tab2:
    st.header("Sensibilités de l'Option (Les Grecques)")
    S = 100.0 
    
    delta, gamma, vega, theta, rho = calculate_call_greeks(S, K, T, r, pred_mlp)
    
    st.subheader("1. Valeurs au point actuel (Modèle MLP)")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Delta (Δ)", f"{delta:.4f}")
    col2.metric("Gamma (Γ)", f"{gamma:.4f}")
    col3.metric("Vega (V)", f"{vega:.4f}")
    col4.metric("Theta (Θ)", f"{theta:.4f}")
    col5.metric("Rho (ρ)", f"{rho:.4f}")
    
    st.divider()
    st.subheader("2. Évolution des Grecques selon le Strike")
    
    strikes_range = np.linspace(70, 130, 50)
    deltas = []
    gammas = []
    
    for strike_val in strikes_range:
        d, g, v, t, rh = calculate_call_greeks(S, strike_val, T, r, pred_mlp)
        deltas.append(d)
        gammas.append(g)
        
    fig_greeks = go.Figure()
    
    fig_greeks.add_trace(go.Scatter(x=strikes_range, y=deltas, mode='lines', name='Delta', yaxis='y1', line=dict(color='blue')))
    fig_greeks.add_trace(go.Scatter(x=strikes_range, y=gammas, mode='lines', name='Gamma', yaxis='y2', line=dict(color='red')))
    
    fig_greeks.update_layout(
        title=f"Delta et Gamma en fonction du Strike (Vol = {pred_mlp:.2%})",
        xaxis_title="Strike (K)",
        yaxis=dict(title="Delta", title_font=dict(color="blue"), tickfont=dict(color="blue")),
        yaxis2=dict(title="Gamma", title_font=dict(color="red"), tickfont=dict(color="red"), overlaying="y", side="right"),
        legend=dict(x=0.01, y=0.99)
    )
    
    st.plotly_chart(fig_greeks, use_container_width=True)

    st.divider()

    st.subheader("3. Surfaces 3D des Grecques (Générées par le modèle MLP)")
    st.markdown(
        "Explorez la forme géométrique spatiale des sensibilités de l'option. "
        "Les Grecques sont calculées dynamiquement sur la grille 3D en utilisant **la volatilité prédite par le réseau de neurones**."
    )

    if mlp_model is not None and scaler_X is not None and scaler_y is not None:
        K_grid_g = np.linspace(70, 130, 40)
        T_grid_g = np.linspace(0.05, 3.0, 40)
        K_mesh_g, T_mesh_g = np.meshgrid(K_grid_g, T_grid_g)
        
        K_flat_g = K_mesh_g.flatten()
        T_flat_g = T_mesh_g.flatten()
        
        vol_base = 0.20
        call_prices_g = black_scholes_call(S0, K_flat_g, T_flat_g, r, vol_base)
        
        df_greeks_input = pd.DataFrame({
            'log-moneyness': np.log(S0 / K_flat_g),
            'Strike Price (K)': K_flat_g,
            'Maturity (T)': T_flat_g,
            'Risk-free Rate (r)': [r] * len(K_flat_g),
            'Call Option Price': call_prices_g
        })
        
        input_g_scaled = scaler_X.transform(df_greeks_input)
        with torch.no_grad():
            preds_g_scaled = mlp_model(torch.tensor(input_g_scaled, dtype=torch.float32)).numpy()
        vol_mlp_grid = scaler_y.inverse_transform(preds_g_scaled).flatten().reshape(40, 40)
        
        delta_mat, gamma_mat, vega_mat, theta_mat, rho_mat = calculate_greeks_grid_vectorized(
            S0, K_mesh_g, T_mesh_g, r, vol_mlp_grid
        )
        
        grecque_choisie = st.radio(
            "**Choisir la surface à afficher :**",
            ["Gamma (Γ)", "Vega (V)", "Delta (Δ)", "Theta (Θ)", "Rho (ρ)"],
            horizontal=True,
            help="Sélectionnez une sensibilité pour mettre à jour la surface 3D interactive."
        )
            
        if "Gamma" in grecque_choisie:
            z_data = gamma_mat
            colorscale = 'Inferno'
            title = "Surface de Gamma (Γ) : Concentration du risque à la monnaie"
            z_label = "Gamma"
        elif "Vega" in grecque_choisie:
            z_data = vega_mat
            colorscale = 'Plasma'
            title = "Surface de Vega (V) : Sensibilité à la volatilité implicite"
            z_label = "Vega"
        elif "Delta" in grecque_choisie:
            z_data = delta_mat
            colorscale = 'Viridis'
            title = "Surface de Delta (Δ) : Probabilité d'exercice dans la monnaie"
            z_label = "Delta"
        elif "Theta" in grecque_choisie:
            z_data = theta_mat
            colorscale = 'Cividis'
            title = "Surface de Theta (Θ) : Érosion du temps (Time Decay)"
            z_label = "Theta (€/jour)"
        else:
            z_data = rho_mat
            colorscale = 'Blues'
            title = "Surface de Rho (ρ) : Sensibilité au taux sans risque"
            z_label = "Rho"

        fig_greeks_3d = go.Figure(data=[go.Surface(
            z=z_data, 
            x=K_grid_g, 
            y=T_grid_g, 
            colorscale=colorscale,
            colorbar=dict(title=z_label, len=0.75)
        )])
        
        fig_greeks_3d.update_layout(
            title=dict(text=title, font=dict(size=18), x=0.5, xanchor='center'), 
            scene=dict(
                xaxis_title='Strike (K)',
                yaxis_title='Maturité T (Années)',
                zaxis_title=z_label,
                aspectratio=dict(x=1.4, y=1.4, z=0.85),
                camera=dict(
                    eye=dict(x=1.6, y=-1.6, z=1.1)
                )
            ),
            margin=dict(l=10, r=10, b=10, t=50),
            height=750 
        )
        
        st.plotly_chart(fig_greeks_3d, use_container_width=True)
            
        if "Gamma" in grecque_choisie:
            st.info("**Analyse:** Observez la formation du pic extrême (le Volcan) lorsque le Strike est proche de 100 (à la monnaie) et que la maturité $T$ se rapproche de $0$. C'est ici que le risque de couverture (hedging risk) est le plus explosif pour un trader d'options.")
        elif "Vega" in grecque_choisie:
            st.info("**Analyse :** Le Vega forme un dôme large qui augmente avec la maturité. Plus une option a du temps avant d'expirer, plus son prix est sensible aux anticipations futures de volatilité du marché.")
with tab3:  

    st.header("Comparaison et Accuracy des modèles")
    
    st.subheader("1. Performances globales (sur le Test Set)")
    st.markdown("Ces métriques représentent la précision historique des modèles évaluée lors de leur entraînement.")
    
    df_metrics = pd.DataFrame({
        "Modèle": ["Random Forest Opt", "XGBoost Opt", "MLP PyTorch"],
        "Score R²": [0.912274, 0.963707, 0.970090],
        "MAE": [0.030320, 0.017171, 0.014698]
    })
    
    st.dataframe(
        df_metrics.style.format({
            "Score R²": "{:.5f}",
            "MAE": "{:.5f}"
        }),
        use_container_width=True,
        hide_index=True
    )
    
    st.divider()        
    if pred_rf is not None and pred_xgb is not None and pred_mlp is not None:
        
        call_rf = black_scholes_call(S0, K, T, r, pred_rf)
        call_xgb = black_scholes_call(S0, K, T, r, pred_xgb)
        call_mlp = black_scholes_call(S0, K, T, r, pred_mlp)
        
        err_rf = abs(call_rf - call_price) 
        err_xgb = abs(call_xgb - call_price) 
        err_mlp = abs(call_mlp - call_price)
        
        df_comp = pd.DataFrame({
            "Modèle": ["Random Forest", "XGBoost", "MLP"],
            "Volatilité Prédite": [pred_rf, pred_xgb, pred_mlp],
            "Call Reconstitué": [call_rf, call_xgb, call_mlp],
            "Erreur de Pricing (%)": [err_rf, err_xgb, err_mlp]
        })
        
        best_model = df_comp.loc[df_comp['Erreur de Pricing (%)'].idxmin(), 'Modèle']
        banniere_meilleur_modele.success(f"Le modèle le plus précis pour cette configuration est le **{best_model}** !")        
        st.subheader("2. Tableau des résultats par modèle")
        st.dataframe(
            df_comp.style.format({
                "Volatilité Prédite": "{:.2%}",
                "Call Reconstitué": "{:.4f} €",
                "Erreur de Pricing (%)": "{:.4f} %"
            }),
            use_container_width=True
        )
        
        st.divider()

        st.subheader("3. Erreur de pricing par modèle")
        fig = px.bar(
            df_comp, 
            x="Modèle", 
            y="Erreur de Pricing (%)", 
            color="Modèle", 
            text_auto='.4f',
        )
        st.plotly_chart(fig, use_container_width=True)
        
    else:
        st.info("Veuillez vous assurer que les 3 modèles sont bien dans le dossier pour afficher la comparaison.")

    st.divider()
    st.subheader("4. Cartographie 3D des Régimes de Domination")
    st.markdown("Chaque point est une option simulée, colorée selon l'algorithme qui l'a prédite avec le moins d'erreur.")

    if rf_model is not None and xgb_model is not None and mlp_model is not None:
        np.random.seed(42)
        sim_n = 1500  
        sim_K = np.random.uniform(70.0, 130.0, size=sim_n)
        sim_T = np.random.uniform(0.1, 3.0, size=sim_n)
        sim_r = np.full(sim_n, r)
        sim_vol = np.random.uniform(0.05, 0.60, size=sim_n)
        
        sim_prices = black_scholes_call(S0, sim_K, sim_T, sim_r, sim_vol)
        
        df_sim = pd.DataFrame({
            'log-moneyness': np.log(S0 / sim_K),
            'Strike Price (K)': sim_K,
            'Maturity (T)': sim_T,
            'Risk-free Rate (r)': sim_r,
            'Call Option Price': sim_prices,
            'True_Vol': sim_vol
        })
        
        preds_rf_sim = rf_model.predict(df_sim.drop(columns=['True_Vol']))
        preds_xgb_sim = xgb_model.predict(df_sim.drop(columns=['True_Vol']))
        
        sim_scaled = scaler_X.transform(df_sim.drop(columns=['True_Vol']))
        with torch.no_grad():
            preds_mlp_sim = scaler_y.inverse_transform(mlp_model(torch.tensor(sim_scaled, dtype=torch.float32)).numpy()).flatten()
            
        df_sim['Err_RF'] = np.abs(df_sim['True_Vol'] - preds_rf_sim)
        df_sim['Err_XGB'] = np.abs(df_sim['True_Vol'] - preds_xgb_sim)
        df_sim['Err_MLP'] = np.abs(df_sim['True_Vol'] - preds_mlp_sim)
        
        df_sim['Meilleur Modèle'] = df_sim[['Err_RF', 'Err_XGB', 'Err_MLP']].idxmin(axis=1).map({
            'Err_RF': 'Random Forest', 'Err_XGB': 'XGBoost', 'Err_MLP': 'PyTorch MLP'
        })
        
        axe_z_choisi = st.radio(
            "**Choisir la dimension de profondeur (Axe Z) :**",
            ["Volatilité Implicite (σ)", "Prix du Call (€)", "Log-Moneyness ln(S0/K)"],
            horizontal=True
        )
        
        if "Volatilité" in axe_z_choisi:
            z_col = 'True_Vol'
            z_label = "Volatilité (σ)"
        elif "Prix" in axe_z_choisi:
            z_col = 'Call Option Price'
            z_label = "Prix du Call (€)"
        else:
            z_col = 'log-moneyness'
            z_label = "Log-Moneyness"

        fig_regime_3d = px.scatter_3d(
            df_sim, 
            x='Strike Price (K)',
            y='Maturity (T)',
            z=z_col,
            color='Meilleur Modèle',
            color_discrete_map={
                'PyTorch MLP': '#EF553B',      
                'XGBoost': '#00CC96',       
                'Random Forest': '#636EFA'   
            },
            opacity=0.85
        )
        
        fig_regime_3d.update_traces(marker=dict(size=4, line=dict(width=0.2, color='white')))
        fig_regime_3d.update_layout(
            title=dict(
                text=f"Domination des modèles dans l'espace (Z = {z_label})",
                font=dict(size=16),
                x=0.5,
                xanchor='center'
            ),
            scene=dict(
                xaxis_title='Strike (K)',
                yaxis_title='Maturité T (Années)',
                zaxis_title=z_label,
                aspectratio=dict(x=1.3, y=1.3, z=0.9),
                camera=dict(eye=dict(x=1.6, y=-1.6, z=0.9))
            ),
            margin=dict(l=0, r=0, b=0, t=50),
            height=700,
            legend=dict(x=0.02, y=0.98, bgcolor='rgba(255,255,255,0.8)'))
        
        st.plotly_chart(fig_regime_3d, use_container_width=True)
        
        st.info("**NB:** En sélectionnant Volatilité Implicite on peut remarquer que les points verts (XGBoost) et bleus (Random Forest) se concentrent souvent dans les recoins extrêmes (faible maturité ou strikes éloignés), là où les modèles d'arbres gèrent mieux les ruptures brutales que le lissage continu du réseau de neurones.")
        
with tab4:
    st.subheader("1. Test de Stabilité (Inputs Plats)")
    st.info("**Hypothèse :** Les prix sont générés avec une volatilité fixe (sigma constante).")
    st.markdown(
        "Surface de Volatilité à Volatilité d'Origine Constante (Scénario de Référence)", 
        help="Test de stabilité : Génération d'une grille de prix avec une volatilité fixe (ex: 20%) pour observer si le modèle reste stable sur l'ensemble de la grille."
    )
    
    if mlp_model is not None:
        K_grid = np.linspace(70, 130, 30)
        T_grid = np.linspace(0.1, 3.0, 30)
        K_mesh, T_mesh = np.meshgrid(K_grid, T_grid)
        
        K_flat = K_mesh.flatten()
        T_flat = T_mesh.flatten()
        
        vol_reference = pred_mlp if pred_mlp is not None else 0.20 

        call_price_grid = black_scholes_call(S0, K_flat, T_flat, r, vol_reference)

        df_surface = pd.DataFrame({
            'log-moneyness': np.log(S0 / K_flat),
            'Strike Price (K)': K_flat,
            'Maturity (T)': T_flat,
            'Risk-free Rate (r)': [r] * len(K_flat),
            'Call Option Price': call_price_grid  
        })
        
        surf_scaled = scaler_X.transform(df_surface)
        surf_tensor = torch.tensor(surf_scaled, dtype=torch.float32)
        with torch.no_grad():
            preds_surf_scaled = mlp_model(surf_tensor).numpy()
        preds_surf = scaler_y.inverse_transform(preds_surf_scaled).flatten()
        
        Vol_matrix = preds_surf.reshape(30, 30)
        
        fig_surf = go.Figure(data=[go.Surface(z=Vol_matrix, x=K_grid, y=T_grid, colorscale='Viridis')])
        fig_surf.update_layout(
            scene=dict(
                xaxis_title='Strike (K)',
                yaxis_title='Maturité (T)',
                zaxis_title='Volatilité Implicite',
            ),
            margin=dict(l=0, r=0, b=0, t=0),
            height=600
        )
        st.plotly_chart(fig_surf, use_container_width=True)

    st.divider()

    st.subheader("2. Reconstitution du Smile de Marché (Vraie Surface vs Prédiction MLP)")
    st.info("**Hypothèse :** Les prix sont générés avec un 'Smile' quadratique et une dynamique de maturité.")
    st.markdown("Ici, nous simulons une 'vraie' surface de volatilité avec un comportement de marché réaliste (Smile et Structure à terme). Nous calculons les prix associés, puis nous demandons au MLP de reconstruire cette surface à partir de ces prix.")
    
    if mlp_model is not None:
        K_grid = np.linspace(70, 130, 30)
        T_grid = np.linspace(0.1, 3.0, 30)
        K_mesh, T_mesh = np.meshgrid(K_grid, T_grid)
        
        K_flat = K_mesh.flatten()
        T_flat = T_mesh.flatten()
        
        true_vol_flat = 0.15 + 0.00015 * (K_flat - S0)**2 + 0.03 * np.exp(-T_flat)
        
        call_price_grid = black_scholes_call(S0, K_flat, T_flat, r, true_vol_flat)

        df_surface = pd.DataFrame({
            'log-moneyness': np.log(S0 / K_flat),
            'Strike Price (K)': K_flat,
            'Maturity (T)': T_flat,
            'Risk-free Rate (r)': [r] * len(K_flat),
            'Call Option Price': call_price_grid
        })
        
        surf_scaled = scaler_X.transform(df_surface)
        surf_tensor = torch.tensor(surf_scaled, dtype=torch.float32)
        with torch.no_grad():
            preds_surf_scaled = mlp_model(surf_tensor).numpy()
        preds_surf = scaler_y.inverse_transform(preds_surf_scaled).flatten()
        
        True_Vol_matrix = true_vol_flat.reshape(30, 30)
        Pred_Vol_matrix = preds_surf.reshape(30, 30)
        
        col_surf1, col_surf2 = st.columns(2)
        
        with col_surf1:
            st.markdown("### Vraie Surface (Réalité)")
            fig_true = go.Figure(data=[go.Surface(z=True_Vol_matrix, x=K_grid, y=T_grid, colorscale='Viridis')])
            fig_true.update_layout(
                scene=dict(
                    xaxis_title='Strike (K)',
                    yaxis_title='Maturité (T)',
                    zaxis_title='Volatilité Implicite',
                ),
                margin=dict(l=0, r=0, b=0, t=0),
                height=500
            )
            st.plotly_chart(fig_true, use_container_width=True)
            
        with col_surf2:
            st.markdown("### Surface Reconstruite par MLP")
            fig_pred = go.Figure(data=[go.Surface(z=Pred_Vol_matrix, x=K_grid, y=T_grid, colorscale='Plasma')])
            fig_pred.update_layout(
                scene=dict(
                    xaxis_title='Strike (K)',
                    yaxis_title='Maturité (T)',
                    zaxis_title='Volatilité Implicite',
                ),
                margin=dict(l=0, r=0, b=0, t=0),
                height=500
            )
            st.plotly_chart(fig_pred, use_container_width=True)



with tab5:
    st.header("Explicabilité du Modèle (XAI)")
    st.markdown("Cette section permet de comprendre la logique interne des modèles, afin de s'assurer que les prédictions ont un sens financier.")
    st.subheader("1. Modèle Random Forest", help="Interprétabilité basée sur les arbres de décision en parallèle (Bagging).")
    if rf_model is not None:
        col_rf1, col_rf2 = st.columns(2)
        
        with col_rf1:
            st.markdown("**Impact des variables (Importance Gini)**")
            importances_rf = rf_model.feature_importances_
            
            df_rf_imp = pd.DataFrame({
                'Variable': input_df.columns,
                'Importance': importances_rf
            }).sort_values(by='Importance', ascending=True)

            fig_rf_imp = px.bar(
                df_rf_imp, 
                x='Importance', 
                y='Variable', 
                orientation='h',
                color='Importance',
                color_continuous_scale='Greys'
            )
            fig_rf_imp.update_layout(showlegend=False, margin=dict(l=0, r=0, t=20, b=0), height=300)
            st.plotly_chart(fig_rf_imp, use_container_width=True)
            
        with col_rf2:
            st.markdown("**Analyse SHAP (Prédiction actuelle)**")
            explainer_rf = shap.TreeExplainer(rf_model)
            shap_values_rf = explainer_rf(input_df)
            
            fig_shap_rf, ax_rf = plt.subplots(figsize=(6, 4))
            shap.plots.waterfall(shap_values_rf[0], show=False)
            
            for text in ax_rf.texts:
                if '$' in text.get_text():
                    text.set_text(text.get_text().replace('$', ''))
            
            plt.tight_layout()
            st.pyplot(fig_shap_rf)
            plt.close(fig_shap_rf)
    else:
        st.warning("Le modèle Random Forest est introuvable pour l'analyse XAI.")

    st.divider()

    st.subheader("2. Modèle XGBRegressor", help="Interprétabilité basée sur le boosting séquentiel d'arbres.")
    if xgb_model is not None:
        colA, colB = st.columns(2)
        
        with colA:
            importances = xgb_model.feature_importances_
            features = input_df.columns
            
            df_imp = pd.DataFrame({
                'Variable': features,
                'Importance': importances
            }).sort_values(by='Importance', ascending=True)

            st.markdown("**Impact des variables (Valeur Absolue)**")
            fig_imp = px.bar(
                df_imp, 
                x='Importance', 
                y='Variable', 
                orientation='h',
                color='Importance',
                color_continuous_scale='Blues'
            )
            fig_imp.update_layout(showlegend=False, margin=dict(l=0, r=0, t=20, b=0), height=300)
            st.plotly_chart(fig_imp, use_container_width=True)
            
        with colB:
            st.markdown("**Analyse SHAP (Prédiction actuelle)**")
            explainer = shap.TreeExplainer(xgb_model)
            shap_values = explainer(input_df)
            
            fig_shap, ax = plt.subplots(figsize=(6, 4))
            shap.plots.waterfall(shap_values[0], show=False)
            
            for text in ax.texts:
                if '$' in text.get_text():
                    text.set_text(text.get_text().replace('$', ''))
            
            plt.tight_layout()
            st.pyplot(fig_shap)
            plt.close(fig_shap)
    else:
        st.warning("Le modèle XGBoost est introuvable. Impossible de générer l'analyse XAI.")

    st.divider()

    st.subheader("3. Modèle PyTorch MLP", help="Interprétabilité par approximation SHAP (DeepExplainer) dédiée aux réseaux de neurones.")
    if mlp_model is not None and scaler_X is not None:
        colC, colD = st.columns(2)
        
        background_tensor = torch.zeros((100, 5), dtype=torch.float32)
        
        explainer_mlp = shap.DeepExplainer(mlp_model, background_tensor)
        
        input_scaled = scaler_X.transform(input_df)
        input_tensor = torch.tensor(input_scaled, dtype=torch.float32)
        
        shap_values_mlp = explainer_mlp.shap_values(input_tensor)
        
        sv_1d = np.array(shap_values_mlp).flatten()
        
        expected_value = explainer_mlp.expected_value
        if isinstance(expected_value, (list, np.ndarray)):
            expected_value = expected_value[0]
            
        with colC:
            st.markdown("**Impact des variables (Valeur Absolue)**")
            df_mlp_imp = pd.DataFrame({
                'Variable': input_df.columns,
                'Impact (SHAP)': np.abs(sv_1d)
            }).sort_values(by='Impact (SHAP)', ascending=True)
            
            fig_mlp_imp = px.bar(
                df_mlp_imp, x='Impact (SHAP)', y='Variable', orientation='h',
                color='Impact (SHAP)', color_continuous_scale='Reds'
            )
            fig_mlp_imp.update_layout(showlegend=False, margin=dict(l=0, r=0, t=0, b=0), height=300)
            st.plotly_chart(fig_mlp_imp, use_container_width=True)

        with colD:
            st.markdown("**Analyse SHAP Locale (Waterfall)**")
            explanation = shap.Explanation(
                values=sv_1d,
                base_values=float(expected_value),
                data=input_df.iloc[0].values,
                feature_names=list(input_df.columns)
            )
            
            fig_shap_mlp, ax = plt.subplots(figsize=(6, 4))
            shap.plots.waterfall(explanation, show=False)
            
            for text in ax.texts:
                if '$' in text.get_text():
                    text.set_text(text.get_text().replace('$', ''))
                    
            plt.tight_layout()
            st.pyplot(fig_shap_mlp)
    else:
        st.warning("Modèle PyTorch ou Scaler introuvable.")