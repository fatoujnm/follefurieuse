import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Charger les données
@st.cache_data
def load_data():
    df = pd.read_csv('AmesHousing.csv')
    df.columns = df.columns.str.strip()  # Nettoyer les noms des colonnes
    return df

df = load_data()

# Prétraitement des données
def preprocess_data(df):
    df.fillna(df.median(numeric_only=True), inplace=True)
    df['Year Built'] = pd.to_numeric(df['Year Built'], errors='coerce')
    df['Age'] = df['Yr Sold'] - df['Year Built']
    df_encoded = pd.get_dummies(df, columns=['Neighborhood', 'House Style'], drop_first=True)
    return df_encoded

df_encoded = preprocess_data(df)

# Sélection des fonctionnalités
features = ['Gr Liv Area', 'Year Built', 'Overall Qual', 'Overall Cond'] + [col for col in df_encoded.columns if col.startswith('Neighborhood_') or col.startswith('House Style_')]
X = df_encoded[features]
y = df_encoded['SalePrice']

# Séparation des données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entraînement du modèle
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# Fonction de prédiction
def predict_price(gr_liv_area, year_built, overall_qual, overall_cond):
    input_data = {
        'Gr Liv Area': gr_liv_area,
        'Year Built': year_built,
        'Overall Qual': overall_qual,
        'Overall Cond': overall_cond,
    }
    for col in features:
        if col not in input_data:
            input_data[col] = 0  # Valeur par défaut pour les colonnes manquantes

    input_df = pd.DataFrame([input_data])
    prediction = model.predict(input_df)
    return prediction[0]

# Interface Utilisateur
st.title("Prédiction du Prix de Vente des Maisons")

# Explications de l'application
st.markdown("""
Cette application prédit le prix de vente des maisons en fonction de leurs caractéristiques. 
Veuillez saisir les informations pertinentes dans la barre latérale pour obtenir une estimation du prix de vente.
""")

st.sidebar.header("Saisir les Informations")

# Saisie des caractéristiques de la maison
gr_liv_area = st.sidebar.slider("Surface Habitable (pieds carrés)", min_value=0, max_value=5000, value=1500)
gr_liv_area_manual = st.sidebar.text_input("Ou entrez la Surface Habitable (pieds carrés) manuellement", value="1500")
if gr_liv_area_manual:
    try:
        gr_liv_area = int(gr_liv_area_manual)
    except ValueError:
        st.sidebar.error("Veuillez entrer une valeur numérique valide pour la Surface Habitable.")

year_built = st.sidebar.slider("Année de Construction", min_value=1900, max_value=2023, value=2000)
year_built_manual = st.sidebar.text_input("Ou entrez l'Année de Construction manuellement", value="2000")
if year_built_manual:
    try:
        year_built = int(year_built_manual)
    except ValueError:
        st.sidebar.error("Veuillez entrer une valeur numérique valide pour l'Année de Construction.")

overall_qual = st.sidebar.slider("Qualité Globale", min_value=1, max_value=10, value=5)
overall_qual_manual = st.sidebar.text_input("Ou entrez la Qualité Globale manuellement", value="5")
if overall_qual_manual:
    try:
        overall_qual = int(overall_qual_manual)
    except ValueError:
        st.sidebar.error("Veuillez entrer une valeur numérique valide pour la Qualité Globale.")

overall_cond = st.sidebar.slider("Condition Globale", min_value=1, max_value=10, value=5)
overall_cond_manual = st.sidebar.text_input("Ou entrez la Condition Globale manuellement", value="5")
if overall_cond_manual:
    try:
        overall_cond = int(overall_cond_manual)
    except ValueError:
        st.sidebar.error("Veuillez entrer une valeur numérique valide pour la Condition Globale.")

# Bouton pour déclencher la prédiction
if st.sidebar.button("Prédire le Prix"):
    predicted_price = predict_price(gr_liv_area, year_built, overall_qual, overall_cond)
    st.write(f"Prix de Vente Prédit: ${predicted_price:,.2f}")

# Visualisations
st.header("Visualisations des Données")

# Distribution du Prix de Vente
st.subheader("Distribution du Prix de Vente")
fig, ax = plt.subplots(figsize=(10, 6))
sns.histplot(df['SalePrice'], kde=True, ax=ax)
ax.set_title('Distribution du Prix de Vente')
ax.set_xlabel('Prix de Vente')
ax.set_ylabel('Fréquence')
ax.legend(['Prix de Vente'])
st.pyplot(fig)

# Relation entre la Surface Habitable et le Prix de Vente
st.subheader("Relation entre la Surface Habitable et le Prix de Vente")
fig, ax = plt.subplots(figsize=(10, 6))
sns.scatterplot(x='Gr Liv Area', y='SalePrice', data=df, ax=ax)
ax.set_title('Relation entre la Surface Habitable et le Prix de Vente')
ax.set_xlabel('Surface Habitable (pieds carrés)')
ax.set_ylabel('Prix de Vente')
ax.legend(['Prix de Vente'])
st.pyplot(fig)

# Relation entre l'Année de Construction et le Prix de Vente
st.subheader("Relation entre l'Année de Construction et le Prix de Vente")
fig, ax = plt.subplots(figsize=(10, 6))
sns.scatterplot(x='Year Built', y='SalePrice', data=df, ax=ax)
ax.set_title('Relation entre l\'Année de Construction et le Prix de Vente')
ax.set_xlabel('Année de Construction')
ax.set_ylabel('Prix de Vente')
ax.legend(['Prix de Vente'])
st.pyplot(fig)

# Relation entre le Quartier et le Prix de Vente
st.subheader("Relation entre le Quartier et le Prix de Vente")
if 'Neighborhood' in df.columns:
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.boxplot(x='Neighborhood', y='SalePrice', data=df, ax=ax)
    ax.set_title('Relation entre le Quartier et le Prix de Vente')
    ax.set_xlabel('Quartier')
    ax.set_ylabel('Prix de Vente')
    ax.tick_params(axis='x', rotation=90)
    ax.legend(['Prix de Vente'])
    st.pyplot(fig)
else:
    st.write("La colonne 'Neighborhood' n'existe pas dans les données après prétraitement.")

# Importance des caractéristiques
st.header("Importance des Caractéristiques")
feature_importances = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
st.bar_chart(feature_importances)
st.write("Les caractéristiques les plus importantes pour prédire le prix de vente des maisons sont affichées ci-dessus.")
