import torch
import torch.nn as nn
import torch.nn.functional as F

class InputFusionNet(nn.Module):
    """
    Opción 1: Fusión a Nivel de Entrada (Input Level Fusion) - CNN 2D
    Esta arquitectura trata la señal EEG como una IMAGEN 2D monocromática
    *Input Shape: (Batch, 1, 21, 128)
    """
    def __init__(self, input_channels=1, num_electrodes=21, time_length=128, base_filters=16):
        """
        Args:
            input_channels (int): 1 
            num_electrodes (int): 21 (Número de electrodos)
            time_length (int): 128 (Longitud temporal de la ventana)
            base_filters (int): 16 (Número de filtros base para las capas convolucionales)
        """
        super(InputFusionNet, self).__init__()
        
        self.base_filters = base_filters
        
        # Extraccion de Características Convolucionales
        
        # Bloque 1: (1, 21, 128) -> (16, ~10, 64)
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=base_filters, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(base_filters)
        self.pool1 = nn.MaxPool2d(kernel_size=2) 
        
        # Bloque 2: (16, ~10, 64) -> (32, ~5, 32)
        self.conv2 = nn.Conv2d(in_channels=base_filters, out_channels=base_filters*2, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(base_filters*2)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        
        # Bloque 3: (32, ~5, 32) -> (64, ~2, 16)
        self.conv3 = nn.Conv2d(in_channels=base_filters*2, out_channels=base_filters*4, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(base_filters*4)
        self.pool3 = nn.MaxPool2d(kernel_size=2)
        
        # Calcular dimensión aplanada después de las capas convolucionales
        with torch.no_grad():
            dummy_input = torch.zeros(1, input_channels, num_electrodes, time_length)
            dummy_output = self._forward_features(dummy_input)
            self.flatten_dim = dummy_output.view(1, -1).size(1)
            
        print(f"Modelo InputFusion inicializado para ({input_channels}x{num_electrodes}x{time_length}).")
        print(f"Features aplanadas antes del clasificador: {self.flatten_dim}")

        # --- CLASIFICADOR ---
        self.fc1 = nn.Linear(self.flatten_dim, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 1) # Salida binaria (Logit)

    def _forward_features(self, x):
        """Pasa los datos solo por la parte convolucional"""
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        return x

    def forward(self, x):
        x = self._forward_features(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

class FeatureFusionNet(nn.Module):
    """
    Opción 2: Fusión a Nivel de Características (Feature Level Fusion) 
    Esta arquitectura procesa cada canal EEG individualmente y luego fusiona las características
    *Input Shape: (Batch, 21, 128)
    """
    def __init__(self, num_electrodes=21, time_length=128):
        """
        Args:
            num_electrodes (int): 21 (Número de electrodos)
            time_length (int): 128 (Longitud temporal de la ventana)
        """
        super(FeatureFusionNet, self).__init__()
        self.num_electrodes = num_electrodes
        self.time_length = time_length
        
        #* Extractor de Características (Compartido para cada canal)
        # Usamos Conv1D porque analizamos una sola señal temporal a la vez
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=8, kernel_size=5, padding=2),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2), # 128 -> 64
            
            nn.Conv1d(in_channels=8, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2), # 64 -> 32
            
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)  # 32 -> 16
        )
        
        # Dimensión de salida del extractor por CADA canal:
        # 32 filtros * 16 muestras temporales = 512 features por canal
        self.features_per_channel = 32 * (time_length // 8) 
        
        # --- Clasificador (Fusion) ---
        # Entrada: Features por canal * Número de canales
        self.fusion_dim = self.features_per_channel * num_electrodes
        
        self.classifier = nn.Sequential(
            nn.Linear(self.fusion_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 1) 
        )

    def forward(self, x):
        # x shape: (Batch, 1, Canales, Tiempo)
        batch_size = x.size(0)
        
        # Reorganizar para procesar cada canal independientemente (Batch * Canales, 1, Tiempo)
        # Quitamos la dimensión 1 (color) que no sirve aquí 
        x = x.squeeze(1) # (Batch, Canales, Tiempo)
        x = x.view(batch_size * self.num_electrodes, 1, self.time_length)
        
        # Extracción de Features (Red Siamese)
        feats = self.feature_extractor(x) # (Batch*Canales, 32, 16)
        
        # Aplanar features individuales 
        feats = feats.view(batch_size * self.num_electrodes, -1) # (Batch*Canales, FeaturesPerChannel)
        
        # Fusión (Concatenación)
        # Volvemos a separar Batch y Canales para concatenarlos lateralmente
        feats = feats.view(batch_size, self.num_electrodes * self.features_per_channel)
        
        # Clasificación
        out = self.classifier(feats)
        
        return out