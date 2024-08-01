import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score

# Titre de l'application
st.title("Analyse et Modélisation des Prix de Vente de Maisons")

# Charger les données
st.header("1. Charger les Données")
@st.cache
def load_data():
    return pd.read_csv('AmesHousing.csv')

df = load_data()
st.write(df.head())

# Afficher les informations sur le DataFrame
st.header("2. Informations sur le DataFrame")
buffer = st.empty()
df_info = df.info(buf=buffer)
st.text(buffer.text)

# Afficher des statistiques descriptives
st.header("3. Statistiques Descriptives")
st.write(df.describe(include='all'))

# Nettoyage des Données
st.header("4. Nettoyage des Données")

# Nettoyer les noms des colonnes en supprimant les espaces et les caractères invisibles
df.columns = df.columns.str.strip()

# Calculer le nombre de valeurs manquantes par colonne
st.subheader("Valeurs Manquantes")
st.write(df.isnull().sum())

# Imputer les valeurs manquantes avec la médiane pour les colonnes numériques
df.fillna(df.median(numeric_only=True), inplace=True)

# Conversion de la colonne 'Year Built' en numérique
df['Year Built'] = pd.to_numeric(df['Year Built'], errors='coerce')

# Détection des Valeurs Aberrantes
st.header("5. Détection des Valeurs Aberrantes")

# Calculer les quartiles et l'IQR pour la colonne 'SalePrice'
Q1 = df['SalePrice'].quantile(0.25)
Q3 = df['SalePrice'].quantile(0.75)
IQR = Q3 - Q1

# Définir les bornes pour les valeurs aberrantes
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Identifier les valeurs aberrantes
outliers = df[(df['SalePrice'] < lower_bound) | (df['SalePrice'] > upper_bound)]
st.subheader("Valeurs Aberrantes")
st.write(outliers)

# Visualisation des valeurs aberrantes pour 'SalePrice' avec un histogramme
st.subheader("Distribution des Prix de Vente")
fig, ax = plt.subplots()
df['SalePrice'].plot(kind='hist', bins=50, edgecolor='k', alpha=0.7, ax=ax)
ax.set_title('Distribution des Prix de Vente')
ax.set_xlabel('Prix de Vente')
ax.set_ylabel('Fréquence')
ax.legend(['SalePrice'])
st.pyplot(fig)

# Ingénierie des Fonctionnalités
st.header("6. Ingénierie des Fonctionnalités")

# Ajouter une fonctionnalité pour la différence d'année de construction
df['Age'] = df['Yr Sold'] - df['Year Built']

# Création de variables indicatrices pour les colonnes catégorielles
df_encoded = pd.get_dummies(df, columns=['Neighborhood', 'House Style'])

st.subheader("DataFrame Encodé")
st.write(df_encoded.head())

# Visualisation de la distribution de SalePrice
st.header("7. Visualisation des Relations")
fig, ax = plt.subplots(figsize=(10, 6))
sns.histplot(df['SalePrice'], kde=True, bins=50, ax=ax)
ax.set_title('Distribution des Prix de Vente')
ax.set_xlabel('Prix de Vente')
ax.set_ylabel('Fréquence')
ax.legend(['SalePrice'])
st.pyplot(fig)

# Relation entre Gr Liv Area et SalePrice
fig, ax = plt.subplots(figsize=(10, 6))
sns.scatterplot(x=df['Gr Liv Area'], y=df['SalePrice'], ax=ax)
ax.set_title('Surface Habitable par Rapport au Prix de Vente')
ax.set_xlabel('Surface Habitable (pieds carrés)')
ax.set_ylabel('Prix de Vente')
ax.legend(['Gr Liv Area vs SalePrice'])
st.pyplot(fig)

# Relation entre Total Bsmt SF et SalePrice
fig, ax = plt.subplots(figsize=(10, 6))
sns.scatterplot(x=df['Total Bsmt SF'], y=df['SalePrice'], ax=ax)
ax.set_title('Surface Totale du Sous-Sol par Rapport au Prix de Vente')
ax.set_xlabel('Surface du Sous-Sol (pieds carrés)')
ax.set_ylabel('Prix de Vente')
ax.legend(['Total Bsmt SF vs SalePrice'])
st.pyplot(fig)

# Relation entre 1st Flr SF et SalePrice
fig, ax = plt.subplots(figsize=(10, 6))
sns.scatterplot(x=df['1st Flr SF'], y=df['SalePrice'], ax=ax)
ax.set_title('Surface du Premier Étage par Rapport au Prix de Vente')
ax.set_xlabel('Surface du Premier Étage (pieds carrés)')
ax.set_ylabel('Prix de Vente')
ax.legend(['1st Flr SF vs SalePrice'])
st.pyplot(fig)

# Matrice de corrélation
st.header("8. Matrice de Corrélation")
fig, ax = plt.subplots(figsize=(12, 10))
corr_matrix = df.corr()
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
ax.set_title('Matrice de Corrélation des Variables Numériques')
st.pyplot(fig)

# Préparation des données pour le modèle de régression
st.header("9. Modélisation")
# Sélection des variables
X = df[['Gr Liv Area', 'Total Bsmt SF', '1st Flr SF']]
y = df['SalePrice']

# Division des données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entraînement du modèle de régression linéaire
model = LinearRegression()
model.fit(X_train, y_train)

# Recherche des paramètres optimaux pour Ridge et Lasso
ridge = Ridge()
lasso = Lasso()

param_grid = {
    'alpha': [0.1, 1.0, 10.0, 100.0]
}

ridge_cv = GridSearchCV(ridge, param_grid, cv=5)
lasso_cv = GridSearchCV(lasso, param_grid, cv=5)

ridge_cv.fit(X_train, y_train)
lasso_cv.fit(X_train, y_train)

# Prédictions avec les modèles optimisés
y_pred_linear = model.predict(X_test)
y_pred_ridge = ridge_cv.predict(X_test)
y_pred_lasso = lasso_cv.predict(X_test)

# Évaluation du modèle
mse_linear = mean_squared_error(y_test, y_pred_linear)
r2_linear = r2_score(y_test, y_pred_linear)

mse_ridge = mean_squared_error(y_test, y_pred_ridge)
r2_ridge = r2_score(y_test, y_pred_ridge)

mse_lasso = mean_squared_error(y_test, y_pred_lasso)
r2_lasso = r2_score(y_test, y_pred_lasso)

st.subheader("Évaluation des Modèles")
st.write(f'Erreur Quadratique Moyenne (MSE) - Régression Linéaire : {mse_linear}')
st.write(f'Coefficient de Détermination (R²) - Régression Linéaire : {r2_linear}')

st.write(f'Erreur Quadratique Moyenne (MSE) - Ridge : {mse_ridge}')
st.write(f'Coefficient de Détermination (R²) - Ridge : {r2_ridge}')

st.write(f'Erreur Quadratique Moyenne (MSE) - Lasso : {mse_lasso}')
st.write(f'Coefficient de Détermination (R²) - Lasso : {r2_lasso}')

# Analyse des résidus pour la régression linéaire
st.header("10. Analyse des Résidus")
fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(y_test, y_pred_linear, label='Prédictions vs Réel')
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4, label='Ligne d\'égalité')
ax.set_xlabel('Valeurs Réelles')
ax.set_ylabel('Valeurs Prédites')
ax.set_title('Valeurs Réelles vs Prédites - Régression Linéaire')
ax.legend()
st.pyplot(fig)
