import pandas as pd
import keras
import numpy as np
import glob
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import SimpleRNN, Dense, Dropout, LSTM
from keras.utils import to_categorical
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import optuna
from keras.models import Sequential

warnings.filterwarnings('ignore')

# Chemin de base pour la recherche des fichiers CSV
chemin_de_base = 'C:/Users/vmellerio/.vscode/Python/'

# Motif pour trouver les dossiers et fichiers contenant '_Vianney_'
pattern_vianney = chemin_de_base + '**/*Experience_Directions*1234*/**/combined_filtered_combined_data*.csv'

# Recherche et chargement des fichiers CSV
fichiers_csv_vianney = glob.glob(pattern_vianney, recursive=True)
dataframes = [pd.read_csv(fichier) for fichier in fichiers_csv_vianney]
data = pd.concat(dataframes)

# Séparation des caractéristiques et des étiquettes
X = data[['Derivative1', 'Derivative2','Sub1','Sub2']].values
y = data['Direction-Blink'].values

# Encodage des étiquettes
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
y_categorical = to_categorical(y_encoded)

# Normalisation des caractéristiques
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_reshaped = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))

# Division en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X_reshaped, y_categorical, test_size=0.2, random_state=42)

def create_model(trial):
    n_lstm_layers = trial.suggest_int('n_lstm_layers', 1, 3)
    dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
    lstm_units = trial.suggest_int('lstm_units', 20, 100)
    optimizer = trial.suggest_categorical('optimizer', ['rmsprop', 'adam', 'sgd'])
    
    model = Sequential()
    model.add(LSTM(lstm_units, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=(n_lstm_layers > 1)))
    model.add(Dropout(dropout_rate))
    
    for i in range(n_lstm_layers - 1):
        model.add(LSTM(lstm_units, return_sequences=(i < n_lstm_layers - 2)))
        model.add(Dropout(dropout_rate))
    
    model.add(Dense(y_categorical.shape[1], activation='softmax'))
    
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

def objective(trial):
    model = create_model(trial)
    
    batch_size = trial.suggest_int('batch_size', 32, 128)
    epochs = trial.suggest_int('epochs', 10, 100)
    
    model.fit(X_train, y_train, validation_split=0.1, epochs=epochs, batch_size=batch_size, verbose=0)
    
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    return accuracy

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)

print('Number of finished trials:', len(study.trials))
print('Best trial:', study.best_trial.params)