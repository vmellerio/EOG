from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.utils import to_categorical
from sklearn.utils import class_weight
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import glob
import os
import re
import time
import warnings

warnings.filterwarnings('ignore')

def process_pattern(base_path, pattern_suffix):
    pattern = os.path.join(base_path, pattern_suffix)
    search_pattern = re.search(r'\*Experience_Directions\*(\w+)\*(\d+)\*(\w+)\*/', pattern)

    results_folder_name = f"Results_Experience_Directions_LSTM_TrainTest{search_pattern.group(1)}_{search_pattern.group(2)}_{search_pattern.group(3) or ''}".rstrip("_") if search_pattern else "Results_pattern"

    csv_files = glob.glob(pattern, recursive=True)
    if not csv_files:
        print(f"No files found for pattern: {pattern}")
        return
    data = pd.concat([pd.read_csv(file) for file in csv_files], ignore_index=True)

    X = data[['Derivative1', 'Derivative2', 'Sub1', 'Sub2']].values
    y = data['Direction-Blink'].values
    X_scaled = StandardScaler().fit_transform(X)
    y_encoded = LabelEncoder().fit_transform(y)
    class_names = LabelEncoder().fit(y).classes_

    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)
    y_train_categorical = to_categorical(y_train)
    y_test_categorical = to_categorical(y_test)

    # Adjust for LSTM input shape
    X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
    X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
    model = Sequential([
        LSTM(100, return_sequences=True, input_shape=(1, X_train.shape[2])),
        Dropout(0.10797610948104297),
        LSTM(100, return_sequences=True),
        Dropout(0.10797610948104297),
        LSTM(100),
        Dropout(0.10797610948104297),
        Dense(y_train_categorical.shape[1], activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    start_time = time.time()
    model.fit(X_train, y_train_categorical, epochs=97, batch_size=67, verbose=2,validation_split=0.1)
    training_time = time.time() - start_time

    # Evaluate the model
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_test_classes = np.argmax(y_test_categorical, axis=1)
    f1 = f1_score(y_test_classes, y_pred_classes, average='weighted')
    # Generate and visualize the confusion matrix
    conf_matrix = confusion_matrix(y_test_classes, y_pred_classes)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', ax=ax, cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix with Class Names')

    # Create results directory if it doesn't exist
    results_directory = os.path.join(base_path, results_folder_name)
    os.makedirs(results_directory, exist_ok=True)

    # Save the confusion matrix and classification report
    confusion_matrix_path = os.path.join(results_directory, "confusion_matrix.jpg")
    plt.savefig(confusion_matrix_path)
    plt.close()

    report = classification_report(y_test_classes, y_pred_classes, target_names=class_names)
    report_path = os.path.join(results_directory, "classification_report.txt")
    with open(report_path, "w") as f:
        f.write(report)

    # Save the trained model
    model_filename = os.path.join(results_directory, "trained_model.h5")
    model.save(model_filename)

    return {
        'Pattern': pattern_suffix,
        'Number of Files': len(csv_files),
        'Total Rows': data.shape[0],
        'Training Time (seconds)': training_time,
        'F1 Score': f1  # Ajout du score F1
    }



# Base path for searching CSV files
base_path = 'C:/Users/vmellerio/.vscode/Python/'

# Patterns to process
patterns = [
    '**/*Experience_Directions*Vianney*1265*Wet*/**/combined_filtered_combined_data*.csv',
    '**/*Experience_Directions*Constance*1265*Wet*/**/combined_filtered_combined_data*.csv',
    '**/*Experience_Directions*TimP*1265*Wet*/**/combined_filtered_combined_data*.csv',
    '**/*Experience_Directions*TimM*1265*Wet*/**/combined_filtered_combined_data*.csv',
    '**/*Experience_Directions*Constance*1234*Wet*/**/combined_filtered_combined_data*.csv',
    '**/*Experience_Directions*Vianney*1234*Wet*/**/combined_filtered_combined_data*.csv',
    '**/*Experience_Directions*Louis*1234*Wet*/**/combined_filtered_combined_data*.csv',
    '**/*Experience_Directions*Carla*1234*Wet*/**/combined_filtered_combined_data*.csv',
    '**/*Experience_Directions*Vianney*5734*Dry*/**/combined_filtered_combined_data*.csv',
    '**/*Experience_Directions*Louis*5734*Dry*/**/combined_filtered_combined_data*.csv',
    '**/*Experience_Directions*Antoinette*5734*Dry*/**/combined_filtered_combined_data*.csv',
    '**/*Experience_Directions*Carla*5734*Dry*/**/combined_filtered_combined_data*.csv',
]

for pattern_suffix in patterns:
    print("Processing pattern:", pattern_suffix)
    process_pattern(base_path, pattern_suffix)

results_summary = [process_pattern(base_path, suffix) for suffix in patterns]
results_df = pd.DataFrame(results_summary)

print("\nTableau récapitulatif des résultats :")
print(results_df)

csv_file_path = os.path.join(base_path, 'results_summary_LSTM_TrainTest_Participants.csv')
results_df.to_csv(csv_file_path, index=False)
print(f"Tableau récapitulatif des résultats sauvegardé dans {csv_file_path}")
