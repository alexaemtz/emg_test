import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob
import os
from scipy.stats import kurtosis, skew

# Cargar los archivos de datos permutados
def load_permuted_files(number):
    data_root = f"F:/orthesis_classification/data/processed/permuted/emg_v0{number}"
    filepaths = glob(os.path.join(data_root, "*.csv"))
    dataframes = [pd.read_csv(fp) for fp in filepaths]
    return dataframes, filepaths

# Calcular la amplitud de Willison
def willison_amplitude(signal, th):
    diff = np.abs(np.diff(signal))
    return np.sum(diff >= th)

# Calcular el cruce por cero
def zero_crossing(signal, th):
    cross = 0
    for i in range(len(signal) - 1):
        prod = signal[i] * signal[i+1]
        diff = np.abs(signal[i] - signal[i+1])
        if prod < 0 and diff > th:
            cross += 1
    return cross

def feature_extraction(dataframes, filepaths, output_dir):
    all_features = []
    print("🔄 Procesando los archivos...")
    
    for df, path in zip(dataframes, filepaths):
        filename = os.path.basename(path).replace(".csv", "")
        print(f"📄 {filename}: {df.shape[0]} muestras temporales, {df.shape[1]} movimientos")
        
        # Procesar las columnas por movimiento
        for movement in df.columns:
            signal = df[movement].values
            
            # Calcular umbral adaptativo (threshold)
            th = np.mean(np.abs(signal)) + 2 * np.std(signal)
            
            # Extraer todas las características
            features = {
                # Identificadores
                "filename": filename,
                "movement": movement,
                "participant": f"v0{number}",
                
                # Características básicas
                "iem": np.sum(np.abs(signal)),                      
                "mean": np.mean(signal),    
                "mav": np.mean(np.abs(signal)),
                "mavs": np.mean(np.abs(np.gradient(np.abs(signal)))),
                "rms": np.sqrt(np.mean(signal**2)),
                "var": np.var(signal),
                "std": np.std(signal),
                
                # Características de forma
                "kurtosis": kurtosis(signal),
                "skewness": skew(signal),
                "max": np.max(signal),
                "min": np.min(signal),
                "range": np.max(signal) - np.min(signal),
                
                # Características de actividad
                "wl": np.sum(np.abs(np.diff(signal))),
                "wamp": willison_amplitude(signal, th),
                "zc": zero_crossing(signal, th),
                "myopulse": np.sum(np.abs(signal) >= th) / len(signal),
                "wamp_norm": willison_amplitude(signal, th) / len(signal),
                
                # Características adicionales
                "ld": np.exp(np.mean(np.log(np.abs(signal) + 1e-10))),
                "aac": np.mean(np.abs(np.diff(signal))),
                "dasdv": np.sqrt(np.mean(np.diff(signal)**2)),
                "ssi": np.sum(signal**2),
                
                # Meta data
                "signal_length": len(signal),
                "sampling_info": f"samples_{len(signal)}"
            }
            
            all_features.append(features)
          
    # Convertir a DataFrame
    features_df = pd.DataFrame(all_features)
        
    output_path = os.path.join(output_dir, f"emg_features_v0{number}.csv")
    features_df.to_csv(output_path, index=False)
        
    # Mostrar resumen del dataset de características
    print(f"\n📊 Resumen del dataset de características:")
    print(f"✅ Total de muestras características: {len(features_df)}")
    print(f"📁 Total de archivos procesados: {len(dataframes)}")
    print(f"🤚 Movimientos únicos: {features_df['movement'].nunique()}")
    print(f"Guardado en {output_path}")

    print(f"\n📈 Distribución de muestras por movimiento: ")
    movement_counts = features_df['movement'].value_counts()
    for movement, count in movement_counts.items():
        print(f"- {movement}: {count} muestras")
        
    return features_df

# Programa principal
if __name__ == "__main__":
    number = int(input("Ingrese el num. del voluntario (0-11): "))
    output_dir = f"F:/orthesis_classification/data/features/"
    os.makedirs(output_dir, exist_ok=True)

    # Cargar datos
    dataframes, filepaths = load_permuted_files(number)
    print(f"📁 Cargados {len(dataframes)} archivos permutados del voluntario {number}")
    
    # Mostrar estructura de ejemplo
    print(f"\n📋 Estructura de datos (ejemplo del primer archivo):")
    print(f"   Forma: {dataframes[0].shape}")
    print(f"   Columnas (movimientos): {list(dataframes[0].columns)}")
    print(f"   Primeras 3 filas:")
    print(dataframes[0].head(3))
    
    # Extraer características por movimiento
    features_df = feature_extraction(dataframes, filepaths, output_dir)
    
    print("✅ Extracción de características completada.")
    print(f"✅ Dataset listo para etiquetado.")
        
        
        
        
        
        
        