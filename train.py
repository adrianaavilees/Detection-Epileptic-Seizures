import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import accuracy_score, roc_auc_score
import numpy as np

from eeg_dataset import EEGSeizureDataset
from models import InputFusionNet

DATA_PATH = '/export/fhome/maed/EpilepsyDataSet/'
MODEL_PATH = 'baseline_input_fusion.pth'  #!!!!!!!!!!!!!!!! Cambia esto si usas otro modelo
BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 10

NUM_CHANNELS = 21   # 21 electrodos
NUM_SAMPLES = 128   # 128 muestras por ventana (128Hz)
INPUT_CHANNELS = 1  # Monocolor (Grayscale)

def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")

    # Cargar Datos
    #* preload_data=True para usar RAM si tienes suficiente, False si prefieres ahorrar memoria
    full_dataset = EEGSeizureDataset(root_dir=DATA_PATH, preload_data=True)
    
    if len(full_dataset) == 0:
        print("No se encontraron datos. Verifica la ruta.")
        return

    # Split Train/Val
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # MODELO OPCION 1: InputFusionNet
    model = InputFusionNet(
        input_channels=INPUT_CHANNELS, 
        num_electrodes=NUM_CHANNELS, 
        time_length=NUM_SAMPLES
    ).to(device)
    
    # Definir Loss y Optimizador
    # PosW (Positive Weight): Útil si hay muchas menos crisis que ventanas normales
    # Aumentarlo (ej. torch.tensor([5.0])) si el dataset está muy desbalanceado
    criterion = nn.BCEWithLogitsLoss() 
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print(f"Iniciando entrenamiento con modelo para Input: ({INPUT_CHANNELS}, {NUM_CHANNELS}, {NUM_SAMPLES})...")
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            
            loss = criterion(outputs, labels.unsqueeze(1))
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
        avg_train_loss = train_loss / len(train_loader)
        
        # Validación
        model.eval()
        val_preds = []
        val_targets = []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                probs = torch.sigmoid(outputs) # Convertir logits a probabilidad [0, 1]
                
                val_preds.extend(probs.cpu().numpy())
                val_targets.extend(labels.cpu().numpy())
        
        # Calcular métricas
        val_preds_bin = [1 if p > 0.5 else 0 for p in val_preds]
        acc = accuracy_score(val_targets, val_preds_bin)
        
        try:
            auc = roc_auc_score(val_targets, val_preds)
        except ValueError:
            auc = 0.5 
            
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg_train_loss:.4f} | Val Acc: {acc:.4f} | Val AUC: {auc:.4f}")

    print("Entrenamiento finalizado")
    # Guardar modelo
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Modelo guardado como {MODEL_PATH}")

if __name__ == "__main__":
    train_model()