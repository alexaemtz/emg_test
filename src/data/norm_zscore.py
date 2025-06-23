# 1.📦 Importar las bibliotecas necesarias
import pandas as pd
import numpy as np

# 2.📂 Cargar el dataset de características
def open_features():
    data_path = "F:/orthesis_classification/data/features/emg_features_labeled.csv"
    try:
        features = pd.read_csv(data_path)
        return features
    except FileNotFoundError:
        print("Archivo no encontrado. Por favor, verifique la ruta de acceso.")
        exit()

# 3.📊 Normalización Z-score
def z_score(df):
    numeric_cols = df.select_dtypes(include=[np.number])
    z_scores = (numeric_cols - numeric_cols.mean()) / numeric_cols.std()
    z_scores.columns = [f"{col}_zcore" for col in numeric_cols.columns]
    return z_scores

# 4.💾 Guardar nuevo dataset con label
def save_z_score(z_score_df, features):
    z_score_df['label'] = features['label'].values
    z_score_df.to_csv('F:/orthesis_classification/data/features/emg_features_labeled_zscore.csv', index=False)

# 5.🚀 Ejecutar
if __name__ == "__main__":
    features = open_features()
    z_score_df = z_score(features)
    save_z_score(z_score_df, features)
    print("✅ El proceso de normalización Z-score ha sido exitoso.")
