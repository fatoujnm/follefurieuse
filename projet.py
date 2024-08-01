import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score

# Création de l'Interface Utilisateur
st.title("Application de Prédiction des Prix de Vente de Maisons")
st.write("""
Cette application web permet aux utilisateurs de saisir les caractéristiques d'une maison et d'obtenir des prédictions sur son prix de vente. 
Elle utilise un modèle de régression linéaire, ainsi que des variantes Ridge et Lasso pour fournir des prévisions précises. 

### Fonctionnalités :
1. **Saisie des Données** : Les utilisateurs peuvent entrer des informations telles que la surface habitable, la surface du sous-sol, et la surface du premier étage.
2. **Prédictions** : Basé sur les données saisies, l'application prédit le prix de vente de la maison en utilisant différents modèles de régression.
3. **Visualisations** : Affichage des graphiques pour aider à comprendre les distributions de données et les relations entre les variables.
4. **Analyse des Résultats** : Les performances des modèles sont évaluées et affichées, fournissant des métriques telles que l'Erreur Quadratique Moyenne (MSE) et le Coefficient de Détermination (R²).

### Mise en Place de la Surveillance :
Nous avons mis en place des outils pour surveiller les performances du modèle en production. Cela inclut :

- **Suivi des Performances** : Surveillance des métriques telles que l'Erreur Quadratique Moyenne (MSE) et le Coefficient de Détermination (R²) pour détecter toute dégradation de la performance.
- **Journalisation des Erreurs** : Mise en place d'un système de journalisation pour enregistrer les erreurs et les anomalies dans les prédictions.
- **Collecte des Retours Utilisateurs** : Recueil des retours des utilisateurs pour identifier des problèmes potentiels et des améliorations nécessaires.
""")

# Charger les données
st.header("1. Charger les Données")
df = pd.read_csv('AmesHousing.csv')
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
st.header("7. Visualisation")
st.subheader("Distribution de SalePrice")
fig, ax = plt.subplots(figsize=(10, 6))
sns.histplot(df['SalePrice'], kde=True, bins=50, ax=ax)
ax.set_title('Distribution des Prix de Vente')
ax.set_xlabel('Prix de Vente')
ax.set_ylabel('Fréquence')
st.pyplot(fig)

# Relation entre Gr Liv Area et SalePrice
st.subheader("Relation entre Gr Liv Area et SalePrice")
fig, ax = plt.subplots(figsize=(10, 6))
sns.scatterplot(x=df['Gr Liv Area'], y=df['SalePrice'], ax=ax)
ax.set_title('Surface Habitable par Rapport au Prix de Vente')
ax.set_xlabel('Surface Habitable (pieds carrés)')
ax.set_ylabel('Prix de Vente')
st.pyplot(fig)

# Relation entre Total Bsmt SF et SalePrice
st.subheader("Relation entre Total Bsmt SF et SalePrice")
fig, ax = plt.subplots(figsize=(10, 6))
sns.scatterplot(x=df['Total Bsmt SF'], y=df['SalePrice'], ax=ax)
ax.set_title('Surface Totale du Sous-Sol par Rapport au Prix de Vente')
ax.set_xlabel('Surface du Sous-Sol (pieds carrés)')
ax.set_ylabel('Prix de Vente')
st.pyplot(fig)

# Relation entre 1st Flr SF et SalePrice
st.subheader("Relation entre 1st Flr SF et SalePrice")
fig, ax = plt.subplots(figsize=(10, 6))
sns.scatterplot(x=df['1st Flr SF'], y=df['SalePrice'], ax=ax)
ax.set_title('Surface du Premier Étage par Rapport au Prix de Vente')
ax.set_xlabel('Surface du Premier Étage (pieds carrés)')
ax.set_ylabel('Prix de Vente')
st.pyplot(fig)

# Filtrage des colonnes numériques pour la matrice de corrélation
numeric_df = df.select_dtypes(include=['number']).dropna(axis=1, how='any')

# Matrice de corrélation
st.subheader("Matrice de Corrélation")
fig, ax = plt.subplots(figsize=(12, 10))
corr_matrix = numeric_df.corr()
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
ax.set_title('Matrice de Corrélation des Variables Numériques')
st.pyplot(fig)

# Préparation des données pour le modèle de régression
st.header("8. Modélisation")

# Sélection des variables
X = df[['Gr Liv Area', 'Total Bsmt SF', '1st Flr SF']]
y = df['SalePrice']

# Division des données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entraînement du modèle de régression linéaire
model = LinearRegression()
model.fit(X_train, y_train)

# Recherche des paramètres optimaux pour Ridge et Lasso
st.subheader("Optimisation des Modèles Ridge et Lasso")

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
st.subheader("Évaluation des Modèles")

mse_linear = mean_squared_error(y_test, y_pred_linear)
r2_linear = r2_score(y_test, y_pred_linear)

mse_ridge = mean_squared_error(y_test, y_pred_ridge)
r2_ridge = r2_score(y_test, y_pred_ridge)

mse_lasso = mean_squared_error(y_test, y_pred_lasso)
r2_lasso = r2_score(y_test, y_pred_lasso)

st.write(f'Erreur Quadratique Moyenne (MSE) - Régression Linéaire : {mse_linear}')
st.write(f'Coefficient de Détermination (R²) - Régression Linéaire : {r2_linear}')

st.write(f'Erreur Quadratique Moyenne (MSE) - Ridge : {mse_ridge}')
st.write(f'Coefficient de Détermination (R²) - Ridge : {r2_ridge}')

st.write(f'Erreur Quadratique Moyenne (MSE) - Lasso : {mse_lasso}')
st.write(f'Coefficient de Détermination (R²) - Lasso : {r2_lasso}')

st.write("""
Pour utiliser cette application, saisissez les informations sur la maison dans la section de saisie des données, et cliquez sur 'Prédire' pour obtenir une estimation du prix de vente basé sur les modèles de régression.
""")

