import pandas as pd
import numpy as np

def open_features():
    try:
        features = pd.read_csv("F:/orthesis_classification/data/features/emg_features_all.csv")
        return features
    except:
        print("No se encontró el archivo de características. Por favor, ejecuta el script 'concat_features.ipynb' para generarlo.")

def labeling(df):
    print("\n🤖Etiquetando el dataset y preparando datos para clasificación...")
    feature_columns = features.select_dtypes(include=[np.number])
    feature_columns = [col for col in feature_columns if col != "signal_length"]
    x = features[feature_columns].values
    y = features["movement"].values
    
    print(f"📊 Matriz de características: {x.shape}")
    print(f"🏷️ Vector de etiquetas: {y.shape}")
    
    return x, y, feature_columns

def save_labeling(x, y, feature_columns):
    print("✅ Guardando etiquetado de datos...")
    features_labeled = pd.DataFrame(x, columns=feature_columns)
    features_labeled["label"] = y
    features_labeled.to_csv("F:/orthesis_classification/data/features/emg_features_labeled.csv", index=False)
    
if __name__ == "__main__":
    features = open_features()
    x, y, feature_columns = labeling(features)
    save_labeling(x, y, feature_columns)
    print("✅ El etiquetado ha sido exitoso y se ha guardado el dataset.")