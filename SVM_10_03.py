import pandas as pd
import glob
import os
import re
import time  # Pour mesurer le temps d'entraînement
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, make_scorer
from sklearn.utils import class_weight
from sklearn.preprocessing import StandardScaler, LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import joblib

# Initialisation de la liste pour stocker les résumés de chaque pattern
results_summary = []

def process_pattern(base_path, pattern_suffix):
    pattern = base_path + pattern_suffix

    # Extraction des parties variables du pattern
    search_pattern = re.search(r'\*Experience_Directions\*(\d+)(?:\*(\w+))?\*/', pattern)
    if search_pattern:
        variable_part1 = search_pattern.group(1)
        variable_part2 = search_pattern.group(2) if search_pattern.group(2) is not None else ""
        results_folder_name = f"Results_Experience_Directions_SVM_10_03{variable_part1}_{variable_part2}".rstrip("_")
    else:
        results_folder_name = "Results_pattern"

    # Recherche et chargement des fichiers CSV
    csv_files = glob.glob(pattern, recursive=True)
    print("Nombre de fichiers traités:", len(csv_files))
    if len(csv_files) == 0:
        print("No files found for pattern:", pattern)
        return
    dataframes = [pd.read_csv(file) for file in csv_files]
    data = pd.concat(dataframes)

    # Séparation des caractéristiques et de l'étiquette
    X = data[['Derivative1', 'Derivative2', 'Sub1', 'Sub2']]
    y = data['Direction-Blink']

    # Normalisation des caractéristiques
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Encoding des étiquettes
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Calculer les poids de classe
    class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_encoded), y=y_encoded)
    class_weights_dict = dict(enumerate(class_weights))
    start_time = time.time()

    # Initialiser le modèle SVC
    model = SVC(C=65000, gamma='auto', kernel='rbf')

    # Mesurer le temps d'entraînement et appliquer la validation croisée
    scores = cross_val_score(model, X_scaled, y_encoded, cv=5, scoring=make_scorer(accuracy_score))
    training_time = time.time() - start_time
    mean_accuracy = np.mean(scores)
    mean_training_time = training_time / len(scores)  # Temps moyen par itération de validation croisée

    # Ajout des résultats dans le tableau récapitulatif
    results_summary.append({
        'Pattern': pattern_suffix,
        'Number of Files': len(csv_files),
        'Total Rows': data.shape[0],
        'Mean Accuracy': mean_accuracy,
        'Mean Training Time (seconds)': mean_training_time,
    })
    print(results_summary)
    # Entraînement du modèle sur l'ensemble des données pour l'évaluation finale
    model.fit(X_scaled, y_encoded)

    model_save_path = os.path.join(base_path, results_folder_name, 'modele_svm_eog_directions.joblib')
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)  # Ensure directory exists
    joblib.dump(model, model_save_path)
    print(f"Modèle sauvegardé dans {model_save_path}")

# Traitement de chaque pattern
base_path = 'C:/Users/vmellerio/.vscode/Python/'
patterns = ['**/*Experience_Directions*1234*Wet*/**/combined_filtered_combined_data*.csv',
            '**/*Experience_Directions*1234*Dry*/**/combined_filtered_combined_data*.csv',
            '**/*Experience_Directions*1265*Wet*/**/combined_filtered_combined_data*.csv',
            '**/*Experience_Directions*1265*Dry*/**/combined_filtered_combined_data*.csv',
            '**/*Experience_Directions*5734*Wet*/**/combined_filtered_combined_data*.csv',
            '**/*Experience_Directions*5734*Dry*/**/combined_filtered_combined_data*.csv',
            '**/*Experience_Directions*5735*Wet*/**/combined_filtered_combined_data*.csv',
            '**/*Experience_Directions*5735*Dry*/**/combined_filtered_combined_data*.csv',
            '**/*Experience_Directions*5737*Wet*/**/combined_filtered_combined_data*.csv',
            '**/*Experience_Directions*5737*Dry*/**/combined_filtered_combined_data*.csv',
            '**/*Experience_Directions*5787*Wet*/**/combined_filtered_combined_data*.csv',
            '**/*Experience_Directions*5787*Dry*/**/combined_filtered_combined_data*.csv',
            '**/*Experience_Directions*6887*Wet*/**/combined_filtered_combined_data*.csv',
            '**/*Experience_Directions*6887*Dry*/**/combined_filtered_combined_data*.csv',]

for pattern_suffix in patterns:
    print("Processing pattern:", pattern_suffix)
    process_pattern(base_path, pattern_suffix)

# Conversion du tableau récapitulatif en DataFrame pour l'afficher et le sauvegarder
results_df = pd.DataFrame(results_summary)

# Afficher le tableau récapitulatif des résultats
print("\nTableau récapitulatif des résultats :")
print(results_df)

# Sauvegarder le tableau récapitulatif des résultats en CSV
csv_file_path = os.path.join(base_path, 'results_summary_SVM_10_03.csv')
results_df.to_csv(csv_file_path, index=False)
print(f"Tableau récapitulatif des résultats sauvegardé dans {csv_file_path}")
