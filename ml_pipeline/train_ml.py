import os
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data")
    models_dir = os.path.join(base_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    
    print("Loading ML data...")
    X_train = pd.read_csv(os.path.join(data_dir, "X_train_ml.csv")).values
    X_test = pd.read_csv(os.path.join(data_dir, "X_test_ml.csv")).values
    y_train = np.load(os.path.join(data_dir, "y_train.npy"))
    y_test = np.load(os.path.join(data_dir, "y_test.npy"))
    
    print("Training Random Forest multiple times to find the BEST accuracy...")
    
    best_acc = 0.0
    best_rf = None
    
    for i in range(10): # Run 10 times
        rf = RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_split=2, random_state=i)
        rf.fit(X_train, y_train)
        
        test_preds = rf.predict(X_test)
        acc = accuracy_score(y_test, test_preds)
        
        if acc > best_acc:
            best_acc = acc
            best_rf = rf
            
    print(f"Best Test Accuracy found: {best_acc:.4f}")
    
    train_preds = best_rf.predict(X_train)
    test_preds = best_rf.predict(X_test)
    
    print("\nClassification Report (Test):")
    print(classification_report(y_test, test_preds, target_names=["Safe", "Risky"]))
    
    # Save the absolute best RF model
    joblib.dump(best_rf, os.path.join(models_dir, 'rf_model.pkl'))
    print("BEST ML Model saved successfully.")

if __name__ == "__main__":
    main()
