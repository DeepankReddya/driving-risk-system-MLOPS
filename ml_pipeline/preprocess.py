import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
import joblib

def create_sequences(data, target, window_size=10):
    """
    Creates sequences for RNN and extracts flat features for ML.
    """
    X_seq, X_ml, y = [], [], []
    
    # Loop over the data to create windows
    for i in range(len(data) - window_size):
        # Sequence for RNN
        window = data[i:(i + window_size)]
        X_seq.append(window)
        
        # Flat features for ML (mean and std of the window)
        ml_features = np.concatenate([np.mean(window, axis=0), np.std(window, axis=0)])
        X_ml.append(ml_features)
        
        # Label (use the label of the last step in the window)
        # Original: 1, 2, 3, 4. 
        # Binary map: 1, 2 -> 0 (Safe), 3, 4 -> 1 (Risky)
        original_label = target.iloc[i + window_size - 1]
        binary_label = 0 if original_label <= 2 else 1
        y.append(binary_label)
        
    return np.array(X_seq), np.array(X_ml), np.array(y)

def main():
    print("Starting Binary Preprocessing...")
    
    # Define paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, "data", "sensor_raw.csv")
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found at {data_path}")
        
    # Load raw data
    df = pd.read_csv(data_path)
    print(f"Loaded data shape: {df.shape}")
    
    # Separate features and target
    target_col = 'Target(Class)'
    features = df.drop(columns=[target_col]).values
    target = df[target_col]
    
    # Scale features globally first
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Create windows
    window_size = 10
    X_seq, X_ml, y = create_sequences(features_scaled, target, window_size)
    print(f"Generated {len(X_seq)} sequences of size {window_size}.")
    print(f"Class distribution - Safe (0): {np.sum(y==0)}, Risky (1): {np.sum(y==1)}")
    
    # Split into train and test
    X_seq_train, X_seq_test, X_ml_train, X_ml_test, y_train, y_test = train_test_split(
        X_seq, X_ml, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Save the processed data
    data_dir = os.path.join(base_dir, "data")
    os.makedirs(data_dir, exist_ok=True)
    
    np.save(os.path.join(data_dir, "X_train_seq.npy"), X_seq_train)
    np.save(os.path.join(data_dir, "X_test_seq.npy"), X_seq_test)
    np.save(os.path.join(data_dir, "y_train.npy"), y_train)
    np.save(os.path.join(data_dir, "y_test.npy"), y_test)
    
    # For ML, save as CSV
    pd.DataFrame(X_ml_train).to_csv(os.path.join(data_dir, "X_train_ml.csv"), index=False)
    pd.DataFrame(X_ml_test).to_csv(os.path.join(data_dir, "X_test_ml.csv"), index=False)
    pd.DataFrame(y_train).to_csv(os.path.join(data_dir, "y_train_ml.csv"), index=False)
    pd.DataFrame(y_test).to_csv(os.path.join(data_dir, "y_test_ml.csv"), index=False)
    
    # Save the scaler for inference
    models_dir = os.path.join(base_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    joblib.dump(scaler, os.path.join(models_dir, "scaler.pkl"))
    
    print("Preprocessing completed successfully!")

if __name__ == "__main__":
    main()
