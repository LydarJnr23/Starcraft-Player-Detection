import pandas as pd
import csv
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# Chemins des fichiers de données
test = 'TEST.CSV'
train = 'TRAIN.CSV'
test_long = 'TEST_LONG.CSV'
train_long = 'TRAIN_LONG.CSV'
sample_sub = 'SAMPLE_SUBMISSION.CSV'

# Fonction pour formater les fichiers d'entraînement
def formatTrain(file):
    profile = []  # Stocke les profils des joueurs
    avatar = []   # Stocke les avatars des joueurs
    moves = []    # Stocke les mouvements des joueurs
    with open(file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',', quotechar="'")
        for row in csv_reader:
            profile.append(row[0])  # Ajoute le profil du joueur
            avatar.append(row[1])   # Ajoute l'avatar du joueur
            moves.append(row[2:])   # Ajoute les mouvements du joueur
    # Crée un DataFrame à partir des données collectées
    d = {'Profile': profile, 'Avatar': avatar, 'Moves': moves}
    df = pd.DataFrame(data=d)
    return df

# Fonction pour formater les fichiers de test
def formatTest(file):
    avatar = []   # Stocke les avatars des joueurs de test
    moves = []    # Stocke les mouvements des joueurs de test
    with open(file) as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',', quotechar="'")
        for row in csvreader:
            avatar.append(row[0])   # Ajoute l'avatar du joueur de test
            moves.append(row[1:])  # Ajoute les mouvements du joueur de test
    # Crée un DataFrame à partir des données collectées
    d = {'Avatar': avatar, 'Moves': moves}
    df = pd.DataFrame(data=d)
    return df

# Nettoyage des données
def clean_data(df):
    # Suppression des valeurs manquantes
    df.dropna(inplace=True)
    
    # Suppression des doublons basés sur la colonne "Moves"
    df['Moves_str'] = df['Moves'].apply(lambda x: ','.join(map(str, x)))
    df.drop_duplicates(subset=['Moves_str'], inplace=True)
    df.drop(columns=['Moves_str'], inplace=True)
    
    # Ajoutez d'autres étapes de nettoyage au besoin
    
    return df

# Extraction des caractéristiques
def extract_features(df):
    features = pd.DataFrame(index=df.index)
    features['num_actions'] = df['Moves'].apply(lambda x: len(x))
    features['unique_actions'] = df['Moves'].apply(lambda x: len(set(x)))
    return features

# Prétraitement des données
def preprocess_data(df_train, df_test):
    # Nettoyage des données
    df_train_cleaned = clean_data(df_train)
    df_test_cleaned = clean_data(df_test)
    
    # Extraction des caractéristiques
    X_train = extract_features(df_train_cleaned)
    X_test = extract_features(df_test_cleaned)
    
    # Encodage des colonnes catégorielles
    le_avatar = LabelEncoder()
    X_train['Avatar'] = le_avatar.fit_transform(df_train_cleaned['Avatar'])
    X_test['Avatar'] = le_avatar.transform(df_test_cleaned['Avatar'])
    
    return X_train, X_test

# Séparation des données en ensembles d'entraînement et de validation
def split_data(X_train, y_train):
    return train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Entraînement du modèle
def train_model(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

# Évaluation du modèle
def evaluate_model(model, X_val, y_val):
    y_pred = model.predict(X_val)
    return accuracy_score(y_val, y_pred)

# Prédiction sur les données de test
def predict_test_data(model, X_test):
    return model.predict(X_test)

# Chargement des données formatées
df_train = formatTrain(train)
df_test = formatTest(test)

# Prétraitement des données
X_train, X_test = preprocess_data(df_train, df_test)

# Encodage des étiquettes des données d'entraînement
le_profile = LabelEncoder()
y_train = le_profile.fit_transform(df_train['Profile'])

# Séparation des données d'entraînement en ensembles d'entraînement et de validation
X_train_split, X_val, y_train_split, y_val = split_data(X_train, y_train)

# Entraînement du modèle
model = train_model(X_train_split, y_train_split)

# Évaluation du modèle
accuracy = evaluate_model(model, X_val, y_val)
print("Précision du modèle sur l'ensemble de validation :", accuracy)

# Prédiction sur les données de test
test_pred = predict_test_data(model, X_test)

# Conversion des prédictions en noms de joueurs
test_pred_player_ids = le_profile.inverse_transform(test_pred)

# Création du fichier de soumission
submission = pd.DataFrame({'prédiction': test_pred_player_ids})
submission.to_csv('SUBMISSION.CSV', index=False)
