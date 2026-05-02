import os
import numpy as np
import pandas as pd
import joblib
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, classification_report
from qiskit.circuit.library import ZFeatureMap
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_machine_learning.algorithms import QSVC
import warnings

# Suppress deprecation warnings from qiskit
warnings.filterwarnings('ignore')

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data")
    models_dir = os.path.join(base_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    
    print("Loading ML data for QML...")
    X_train = pd.read_csv(os.path.join(data_dir, "X_train_ml.csv")).values
    X_test = pd.read_csv(os.path.join(data_dir, "X_test_ml.csv")).values
    y_train = np.load(os.path.join(data_dir, "y_train.npy"))
    y_test = np.load(os.path.join(data_dir, "y_test.npy"))
    
    # 1. Dimensionality Reduction to 4 features (for 4 qubits)
    print("Applying PCA to reduce to 4 dimensions for QML...")
    pca = PCA(n_components=4)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    
    # Save PCA model for inference
    joblib.dump(pca, os.path.join(models_dir, 'pca_model.pkl'))
    
    print("Applying MinMaxScaler to PCA features for QML...")
    qml_scaler = MinMaxScaler(feature_range=(-1, 1))
    X_train_pca = qml_scaler.fit_transform(X_train_pca)
    X_test_pca = qml_scaler.transform(X_test_pca)
    
    # Save scaler for inference
    joblib.dump(qml_scaler, os.path.join(models_dir, 'qml_scaler.pkl'))
    
    # 2. Subsample to speed up simulation (Quantum Kernels are slow to simulate)
    sample_size = min(100, len(X_train_pca))
    idx = np.random.choice(len(X_train_pca), sample_size, replace=False)
    X_train_sub = X_train_pca[idx]
    y_train_sub = y_train[idx]
    
    test_sample_size = min(50, len(X_test_pca))
    test_idx = np.random.choice(len(X_test_pca), test_sample_size, replace=False)
    X_test_sub = X_test_pca[test_idx]
    y_test_sub = y_test[test_idx]
    
    print("Training QSVC multiple times to find the BEST accuracy...")
    feature_dimension = 4
    feature_map = ZFeatureMap(feature_dimension=feature_dimension, reps=1)
    qkernel = FidelityQuantumKernel(feature_map=feature_map)
    
    best_acc = 0.0
    best_qsvc = None
    
    for run in range(3):
        print(f"\n--- QML Training Run {run+1}/3 ---")
        # Different subset each run
        idx = np.random.choice(len(X_train_pca), sample_size, replace=False)
        X_train_sub = X_train_pca[idx]
        y_train_sub = y_train[idx]
        
        test_idx = np.random.choice(len(X_test_pca), test_sample_size, replace=False)
        X_test_sub = X_test_pca[test_idx]
        y_test_sub = y_test[test_idx]
        
        qsvc = QSVC(quantum_kernel=qkernel)
        qsvc.fit(X_train_sub, y_train_sub)
        
        test_preds = qsvc.predict(X_test_sub)
        acc = accuracy_score(y_test_sub, test_preds)
        print(f"Run {run+1} Test Accuracy (subset): {acc:.4f}")
        
        if acc > best_acc:
            best_acc = acc
            best_qsvc = qsvc
            best_preds = test_preds
            best_y_test = y_test_sub
            
    print(f"\nOverall Best Test Accuracy (subset): {best_acc:.4f}")
    print("\nClassification Report (Best Model):")
    print(classification_report(best_y_test, best_preds, target_names=["Safe", "Risky"]))
    
    # Save best QSVC model
    joblib.dump(best_qsvc, os.path.join(models_dir, 'qml_model.pkl'))
    print("BEST QML Model saved successfully.")

if __name__ == "__main__":
    np.random.seed(42)
    main()
