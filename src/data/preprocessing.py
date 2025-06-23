import pandas as pd
import numpy as np
from scipy.signal import butter, sosfilt
import os
import itertools

# Variables para el filtrado de los datos (Envolvente)
cutoff = 10.0
fs = 1000.0

# Cargar el archivo de EMG en formato bits
def load_bit_emg(filename):
    path = f"data/raw/bit_signals/tabledata{filename}.txt"
    raw_emg_bits = pd.read_csv(path)
    return raw_emg_bits

# Convertir la señal de bits a mV
def convert_to_mv(bits_signal, adc_resolution=1023.0, vref_mv=5000.0):
    bits_raw_signal = (bits_signal / adc_resolution) * vref_mv
    return bits_raw_signal

# Guardar el DataFrame convertido a mv como CSV
def save_emg_file_mv(raw_emg, number):
    path = f"data/raw/mv_signals/emg_v0{number}.csv"
    pd.DataFrame(raw_emg).to_csv(path, index=False)

# Filtrar el DataFrame para obtener las envolventes del EMG
def lowpass_filter(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    sos = butter(order, normal_cutoff, btype="low", analog=False, output="sos")
    return sos

def apply_lowpass_filter(raw_emg, cutoff, fs, order=5):
    sos = lowpass_filter(cutoff, fs, order)
    # Si es un DataFrame, aplicar el filtro por columna conservando nombres
    if isinstance(raw_emg, pd.DataFrame):
        filtered = raw_emg.copy()
        for col in filtered.columns:
            filtered[col] = sosfilt(sos, filtered[col])
        return filtered
    else:
        # Si es una serie o array, convertirlo a DataFrame con columna "emg"
        return pd.DataFrame(sosfilt(sos, raw_emg), columns=["emg"])

# Guardar la señal filtrada
def save_filtered_emg(filtered_emg, number):
    path = f"data/processed/filtered/emg_v0{number}_filtered.csv"
    filtered_emg.to_csv(path, index=False)

# Segmentar la señal en ventanas fijas
def segment_signal(filtered_emg, segment_length):
    segments = []
    total_samples = len(filtered_emg)
    
    while total_samples >= segment_length:
        segment = filtered_emg[:segment_length]      # Tomar las primeras muestras segment_length 
        segments.append(segment)                     # Agregar al listado de segmentos
        filtered_emg = filtered_emg[segment_length:] # Cortar la señal
        total_samples = len(filtered_emg)            # Actualizar el total
        
    return segments

# Guardar los segmentos por separado
def save_segmented_emg(segments, number):
    folder = f"data/processed/segmented/emg_v0{number}"
    os.makedirs(folder, exist_ok=True)
    
    for idx, seg in enumerate(segments):
        path = f"{folder}/segment_{idx}.csv"
        pd.DataFrame(seg).to_csv(path, index=False)

# Permutar los segmentos (factorial de la cantidad de segmentos)
def generate_permutations(segments):
    return list(itertools.permutations(segments))

# Guardar cada permutación en un archivo individual
def save_permuted_segments(permutations, number):
    folder = f"data/processed/permuted/emg_v0{number}"
    os.makedirs(folder, exist_ok=True)
    
    for idx, permutation in enumerate(permutations):
        combined = pd.concat(permutation, ignore_index=True)
        path = f"{folder}/permutation_{idx}.csv"
        combined.to_csv(path, index=False)

# Programa principal
if __name__ == "__main__":
    filename = input("Ingrese el nombre del paciente: ")
    number = input(str("Ingrese el número de voluntario: "))

    # 1. Cargar el archivo EMG de señal cruda en formato bits
    raw_emg_bits = load_bit_emg(filename)
    
    # 2. Convertir la señal de bits a mV y guardar el archivo en CSV
    raw_emg = convert_to_mv(raw_emg_bits)
    save_emg_file_mv(raw_emg, number)
    
    # 3. Filtrar la señal de EMG para obtener las envolventes y guardar EMG filtrado
    filtered_emg = apply_lowpass_filter(raw_emg, cutoff=cutoff, fs=fs, order=5)
    save_filtered_emg(filtered_emg, number)
    
    # 4. Segmentar la señal de EMG filtrada y guardar los segmentos
    segments = segment_signal(filtered_emg, segment_length=1500)
    save_segmented_emg(segments, number)
    
    # 5. Generar las 24 permutaciones (si hay 4 segmentos) y guardarlas como archivos individuales
    permutations = generate_permutations(segments)
    save_permuted_segments(permutations, number)
