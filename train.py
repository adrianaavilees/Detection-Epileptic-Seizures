import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_curve, auc, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np
import os

from eeg_dataset import EEGSeizureDataset
from models import InputFusionNet, FeatureFusionNet

# --- CONFIGURACIÓN ---
DATA_PATH = '/export/fhome/maed/EpilepsyDataSet/'
K_FOLDS = 5            
BATCH_SIZE = 64
EPOCHS_PER_FOLD = 5    
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

OUTPUT_DIR = "models" 
os.makedirs(OUTPUT_DIR, exist_ok=True)

def calculate_metrics(y_true, y_probs, threshold=0.5):
    """Calcula métricas binarias basadas en un umbral predefinido"""
    y_pred = (np.array(y_probs) > threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    
    acc = accuracy_score(y_true, y_pred)
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0  # Recall Positivo
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0  # Recall Negativo
    
    return acc, sensitivity, specificity

def run_cross_validation_complete(model_class, dataset, model_name="Model"):
    print(f"\n--- Procesando {model_name} ---")
    
    groups = [s['patient'] for s in dataset.samples_index]
    labels = [s['label'] for s in dataset.samples_index]
    indices = np.arange(len(dataset))
    
    gkf = GroupKFold(n_splits=K_FOLDS)
    
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    
    metrics_per_fold = {'accuracy': [], 'sensitivity': [], 'specificity': []}
    y_true_all = []
    y_pred_all = []  
    
    for fold, (train_idx, val_idx) in enumerate(gkf.split(indices, labels, groups=groups)):
        print(f"  Fold {fold+1}/{K_FOLDS}...", end=" ", flush=True)
        
        train_sub = Subset(dataset, train_idx)
        val_sub = Subset(dataset, val_idx)
        
        train_loader = DataLoader(train_sub, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_sub, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
        
        # Modelo 
        model = model_class(num_electrodes=21, time_length=128).to(DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.BCEWithLogitsLoss() 
        
        # Train
        model.train()
        for epoch in range(EPOCHS_PER_FOLD):
            for x, y in train_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                optimizer.zero_grad()
                out = model(x)
                loss = criterion(out, y.unsqueeze(1))
                loss.backward()
                optimizer.step()
        
        # GUARDAR EL MODELO
        clean_name = model_name.replace(" ", "_").replace("(", "").replace(")", "")
        save_path = os.path.join(OUTPUT_DIR, f"{clean_name}_fold{fold+1}.pth")
        torch.save(model.state_dict(), save_path)
        print(f"Modelo guardado: {save_path}")
        
        # Evaluación
        model.eval()
        fold_y_true = []
        fold_y_scores = []
        
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                out = model(x)
                probs = torch.sigmoid(out)
                fold_y_true.extend(y.cpu().numpy())
                fold_y_scores.extend(probs.cpu().numpy())
        
        # Métricas
        if len(np.unique(fold_y_true)) < 2:
            continue
            
        fpr, tpr, _ = roc_curve(fold_y_true, fold_y_scores)
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        
        acc, sens, spec = calculate_metrics(fold_y_true, fold_y_scores)
        metrics_per_fold['accuracy'].append(acc)
        metrics_per_fold['sensitivity'].append(sens)
        metrics_per_fold['specificity'].append(spec)
        
        y_true_all.extend(fold_y_true)
        y_pred_all.extend((np.array(fold_y_scores) > 0.5).astype(int))
        
        print(f"AUC: {roc_auc:.3f}")

    # Estadísticas Finales
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    std_tpr = np.std(tprs, axis=0)
    
    stats = {
        'mean_acc': np.mean(metrics_per_fold['accuracy']), 'std_acc': np.std(metrics_per_fold['accuracy']),
        'mean_sens': np.mean(metrics_per_fold['sensitivity']), 'std_sens': np.std(metrics_per_fold['sensitivity']),
        'mean_spec': np.mean(metrics_per_fold['specificity']), 'std_spec': np.std(metrics_per_fold['specificity']),
        'mean_auc': mean_auc, 'std_auc': std_auc
    }
    
    return {
        'mean_fpr': mean_fpr, 'mean_tpr': mean_tpr, 'std_tpr': std_tpr,
        'stats': stats, 'y_true_all': y_true_all, 'y_pred_all': y_pred_all, 'name': model_name
    }

def plot_roc_and_tables(res1, res2, save_path='comparison_roc_curve_test.png'):
    """Genera imagen ROC y Tabla para cada modelo por separado"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # OPCIÓN 1 (Input Fusion) 
    
    # ROC Curve (Arriba Izquierda)
    ax_roc1 = axes[0, 0]
    ax_roc1.plot([0, 1], [0, 1], linestyle='--', lw=2, color='gray', alpha=.5)
    ax_roc1.plot(res1['mean_fpr'], res1['mean_tpr'], color='blue', lw=2,
                 label=f"AUC = {res1['stats']['mean_auc']:.2f} $\pm$ {res1['stats']['std_auc']:.2f}")
    
    tprs_upper1 = np.minimum(res1['mean_tpr'] + res1['std_tpr'], 1)
    tprs_lower1 = np.maximum(res1['mean_tpr'] - res1['std_tpr'], 0)
    ax_roc1.fill_between(res1['mean_fpr'], tprs_lower1, tprs_upper1, color='blue', alpha=.2, label=r'$\pm$ 1 std. dev.')
    
    ax_roc1.set_title(f"ROC Curve: {res1['name']}", fontsize=14, fontweight='bold')
    ax_roc1.set_xlabel('False Positive Rate')
    ax_roc1.set_ylabel('True Positive Rate')
    ax_roc1.legend(loc="lower right")
    ax_roc1.grid(alpha=0.3)
    
    # Tabla Métricas (Arriba Derecha)
    ax_tab1 = axes[0, 1]
    ax_tab1.axis('off')
    ax_tab1.set_title(f"Métricas Detalladas: {res1['name']}", fontsize=14, fontweight='bold')
    
    rows1 = ['Accuracy', 'AUC', 'Recall Pos (Sens)', 'Recall Neg (Spec)']
    text1 = [
        [f"{res1['stats']['mean_acc']:.3f} ± {res1['stats']['std_acc']:.3f}"],
        [f"{res1['stats']['mean_auc']:.3f} ± {res1['stats']['std_auc']:.3f}"],
        [f"{res1['stats']['mean_sens']:.3f} ± {res1['stats']['std_sens']:.3f}"],
        [f"{res1['stats']['mean_spec']:.3f} ± {res1['stats']['std_spec']:.3f}"]
    ]
    
    tab1 = ax_tab1.table(cellText=text1, rowLabels=rows1, colLabels=['Mean ± Std Dev'], 
                         loc='center', cellLoc='center', colWidths=[0.5])
    tab1.auto_set_font_size(False)
    tab1.set_fontsize(14)
    tab1.scale(1, 2)

    # OPCIÓN 2 (Feature Fusion) 
    
    # ROC Curve (Abajo Izquierda)
    ax_roc2 = axes[1, 0]
    ax_roc2.plot([0, 1], [0, 1], linestyle='--', lw=2, color='gray', alpha=.5)
    ax_roc2.plot(res2['mean_fpr'], res2['mean_tpr'], color='red', lw=2,
                 label=f"AUC = {res2['stats']['mean_auc']:.2f} $\pm$ {res2['stats']['std_auc']:.2f}")
    
    tprs_upper2 = np.minimum(res2['mean_tpr'] + res2['std_tpr'], 1)
    tprs_lower2 = np.maximum(res2['mean_tpr'] - res2['std_tpr'], 0)
    ax_roc2.fill_between(res2['mean_fpr'], tprs_lower2, tprs_upper2, color='red', alpha=.2, label=r'$\pm$ 1 std. dev.')
    
    ax_roc2.set_title(f"ROC Curve: {res2['name']}", fontsize=14, fontweight='bold')
    ax_roc2.set_xlabel('False Positive Rate')
    ax_roc2.set_ylabel('True Positive Rate')
    ax_roc2.legend(loc="lower right")
    ax_roc2.grid(alpha=0.3)
    
    # Tabla Métricas (Abajo Derecha)
    ax_tab2 = axes[1, 1]
    ax_tab2.axis('off')
    ax_tab2.set_title(f"Métricas Detalladas: {res2['name']}", fontsize=14, fontweight='bold')
    
    rows2 = ['Accuracy', 'AUC', 'Recall Pos (Sens)', 'Recall Neg (Spec)']
    text2 = [
        [f"{res2['stats']['mean_acc']:.3f} ± {res2['stats']['std_acc']:.3f}"],
        [f"{res2['stats']['mean_auc']:.3f} ± {res2['stats']['std_auc']:.3f}"],
        [f"{res2['stats']['mean_sens']:.3f} ± {res2['stats']['std_sens']:.3f}"],
        [f"{res2['stats']['mean_spec']:.3f} ± {res2['stats']['std_spec']:.3f}"]
    ]
    
    tab2 = ax_tab2.table(cellText=text2, rowLabels=rows2, colLabels=['Mean ± Std Dev'], 
                         loc='center', cellLoc='center', colWidths=[0.5])
    tab2.auto_set_font_size(False)
    tab2.set_fontsize(14)
    tab2.scale(1, 2)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"Imagen 1 guardada: {save_path}")

def plot_confusion_matrices(res1, res2, save_path='comparison_matrius_test.png'):
    """Genera imagen matrices de confusión acumuladas lado a lado"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Matriz 1
    cm1 = confusion_matrix(res1['y_true_all'], res1['y_pred_all'], labels=[0, 1])
    disp1 = ConfusionMatrixDisplay(confusion_matrix=cm1, display_labels=['Normal', 'Crisis'])
    disp1.plot(ax=axes[0], cmap='Blues', values_format='d', colorbar=False)
    axes[0].set_title(f"Matriz Acumulada: {res1['name']}", fontsize=14, fontweight='bold')
    
    # Matriz 2
    cm2 = confusion_matrix(res2['y_true_all'], res2['y_pred_all'], labels=[0, 1])
    disp2 = ConfusionMatrixDisplay(confusion_matrix=cm2, display_labels=['Normal', 'Crisis'])
    disp2.plot(ax=axes[1], cmap='Reds', values_format='d', colorbar=False)
    axes[1].set_title(f"Matriz Acumulada: {res2['name']}", fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"Imagen 2 guardada: {save_path}")

def main():
    print(f"Usando dispositivo: {DEVICE}")
    print("Cargando dataset...")
    full_dataset = EEGSeizureDataset(root_dir=DATA_PATH, preload_data=True)
    
    if len(full_dataset) == 0:
        print("Error: Dataset vacío.")
        return

    # Ejecutar CV
    res1 = run_cross_validation_complete(InputFusionNet, full_dataset, "Opción 1 (Input)")
    res2 = run_cross_validation_complete(FeatureFusionNet, full_dataset, "Opción 2 (Feature)")
    
    # Generar las dos imagenes
    print("\nGenerando gráficas...")
    plot_roc_and_tables(res1, res2)
    plot_confusion_matrices(res1, res2)
    print("\nEntrenamiento y evaluación completados")

if __name__ == "__main__":
    main()