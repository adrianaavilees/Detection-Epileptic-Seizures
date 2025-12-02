# archivos .npz (señal) y .parquet (metadata) por paciente
# Asegurar numero de canales consistente (21-23 segun el pdf) 
# Si algun paciente tiene un numero de canales diferente, rellenar (padding) o intereseccion (selecionar canales comunes)

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import glob
import time
from collections import defaultdict


class EEGSeizureDataset(Dataset):
    """
    Dataset para cargar ventanas de EEG desde archivos .npz y metadatos .parquet
    """
    def __init__(self, root_dir, transform=None, preload_data=False):
        self.root_dir = root_dir
        self.transform = transform
        self.samples_index = [] # Lista de tuplas (path_npz, index_in_npz, label)
        self.preload_data = preload_data # Si True, cargar todos los datos en RAM al inicio
        self.cached_data = [] # Aquí se guardarán los datos si preload_data=True
        
        all_files = sorted(glob.glob(os.path.join(root_dir, "*_metadata_*.parquet")))
        
        print(f"Indexando dataset desde {root_dir}...")
        
        # Crear el indice de muestras 
        for meta_path in all_files:
            # Extraer ID del paciente (ej. chb01) del nombre del archivo
            filename = os.path.basename(meta_path)
            patient_id = filename.split('_')[0]
                
            # Construir path al archivo de ventanas correspondiente (.npz)
            # chb01_seizure_metadata_1.parquet -> chb01_seizure_EEGwindow_1.npz
            base_name = filename.replace("_metadata_", "_EEGwindow_").replace(".parquet", ".npz")
            npz_path = os.path.join(root_dir, base_name)
            
            if not os.path.exists(npz_path):
                print(f"WARNING: No se encontró {npz_path}, saltando...")
                continue
            
            # Leer metadata para saber cuantas ventanas hay y sus etiquetas
            try:
                df = pd.read_parquet(meta_path)
                #? Assumimos parquet tiene una fila por ventana en el npz y en el mismo orden
                # 'class' (0 o 1)
                labels = df['class'].values.astype(int)
                
                for idx, label in enumerate(labels):
                    self.samples_index.append({
                        'npz_path': npz_path,
                        'idx': idx,
                        'label': label
                    })
                    
            except Exception as e:
                print(f"Error leyendo {meta_path}: {e}")

        print(f"Dataset indexado: {len(self.samples_index)} ventanas totales encontradas")

        # Cargar todos los datos en RAM si se especifica
        if self.preload_data:
            print("Iniciando carga masiva a RAM...")
            start_time = time.time()
            
            # Agrupamos por archivo para no abrir y cerrar el mismo .npz (optimiza la lectura)
            file_to_indices = defaultdict(list)
            
            for i, sample in enumerate(self.samples_index):
                file_to_indices[sample['npz_path']].append((sample['idx'], i))
            
            # Inicializamos la lista de cache con None
            self.cached_data = [None] * len(self.samples_index)
            
            for npz_path, indices_list in file_to_indices.items():
                try:
                    with np.load(npz_path, allow_pickle=True) as data:
                        key = data.files[0]
                        full_array = data[key] # Cargamos el array entero del archivo una sola vez
                        
                        for local_idx, global_idx in indices_list:
                            # Extraemos la ventana específica
                            signal = full_array[local_idx].astype(np.float32)
                            
                            # Normalización Z-Score inmediata para ahorrar cómputo luego
                            mean = np.mean(signal, axis=1, keepdims=True)
                            std = np.std(signal, axis=1, keepdims=True) + 1e-6
                            signal = (signal - mean) / std
                            
                            # Convertir a Tensor y añadir dimensión de canal (1, C, T)
                            signal_tensor = torch.from_numpy(signal).unsqueeze(0)
                            
                            self.cached_data[global_idx] = signal_tensor
                            
                except Exception as e:
                    print(f"Error cargando bloque {npz_path}: {e}")

            print(f"Carga completa en {time.time() - start_time:.2f} segundos")

    def __len__(self):
        return len(self.samples_index)

    def __getitem__(self, idx):
        sample_info = self.samples_index[idx]
        label = torch.tensor(sample_info['label'], dtype=torch.float32)
        
        # RAM (Rápido pero usa más memoria)
        if self.preload_data:
            signal_window = self.cached_data[idx]
            if signal_window is None: # Fallback por si fallo la carga
                return torch.zeros(1, 1, 1), label
            return signal_window, label

        # DISCO (Lento pero ahorra memoria)
        npz_path = sample_info['npz_path']
        window_idx = sample_info['idx']
        
        try:
            with np.load(npz_path, allow_pickle=True) as data:
                key = data.files[0] 
                signal_window = data[key][window_idx].astype(np.float32)

            if self.transform:
                signal_window = self.transform(signal_window)
            else:
                mean = np.mean(signal_window, axis=1, keepdims=True)
                std = np.std(signal_window, axis=1, keepdims=True) + 1e-6
                signal_window = (signal_window - mean) / std

            signal_window = torch.from_numpy(signal_window).unsqueeze(0) 
            return signal_window, label
            
        except Exception:
            return torch.zeros(1, 1, 1), label