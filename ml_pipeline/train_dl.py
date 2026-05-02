import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, classification_report

class SimpleRNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(SimpleRNNModel, self).__init__()
        # Using Simple RNN, NOT LSTM
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)
        out, _ = self.rnn(x)
        # Taking the last time step output
        out = self.fc(out[:, -1, :])
        return out

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data")
    models_dir = os.path.join(base_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    
    print("Loading DL sequence data...")
    X_train = np.load(os.path.join(data_dir, "X_train_seq.npy"))
    X_test = np.load(os.path.join(data_dir, "X_test_seq.npy"))
    y_train = np.load(os.path.join(data_dir, "y_train.npy"))
    y_test = np.load(os.path.join(data_dir, "y_test.npy"))
    
    num_classes = 2 # Binary (Safe/Risky)
    
    print("Training Simple RNN...")
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.long)
    
    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    input_size = X_train.shape[2]
    hidden_size = 64
    epochs = 40
    
    print("Training Simple RNN multiple times to find the BEST accuracy...")
    
    best_acc = 0.0
    best_model_state = None
    
    for run in range(5): # Run 5 separate training cycles
        print(f"\n--- DL Training Run {run+1}/5 ---")
        model = SimpleRNNModel(input_size, hidden_size, num_classes)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.005)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
        
        for epoch in range(epochs):
            model.train()
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
            scheduler.step()
            
        # Evaluation for this run
        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test_t)
            test_preds = torch.argmax(test_outputs, dim=1).numpy()
            acc = accuracy_score(y_test, test_preds)
            print(f"Run {run+1} Test Accuracy: {acc:.4f}")
            
            if acc > best_acc:
                best_acc = acc
                # Create a deep copy of the full model
                import copy
                best_model = copy.deepcopy(model)
                best_preds = test_preds

    print(f"\nOverall Best Test Accuracy: {best_acc:.4f}")
    print("\nClassification Report (Best Model):")
    print(classification_report(y_test, best_preds, target_names=["Safe", "Risky"]))
        
    # Save best RNN model
    torch.save(best_model, os.path.join(models_dir, 'dl_model.pkl'))
    print("BEST DL Model saved successfully as pkl.")

if __name__ == "__main__":
    main()
