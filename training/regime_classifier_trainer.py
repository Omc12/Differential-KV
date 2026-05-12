"""
training/regime_classifier_trainer.py

Trains a deep regime classifier using collected latent trajectory features.
Targets >85% accuracy.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json
import os
import numpy as np
from typing import List, Dict, Any

class RegimeModel(nn.Module):
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )
        
    def forward(self, x):
        return self.net(x)

class RegimeDataset(Dataset):
    def __init__(self, data_path: str):
        with open(data_path, "r") as f:
            raw_data = json.load(f)
            
        self.regimes = [
            "mathematical_reasoning",
            "code_generation",
            "recursive_planning",
            "tool_use_chains",
            "retrieval_heavy",
            "narrative_dialogue",
            "mixed_mode"
        ]
        self.regime_to_idx = {r: i for i, r in enumerate(self.regimes)}
        
        self.X = []
        self.y = []
        
        # Define feature keys
        self.feature_keys = [
            "latent_drift", "curvature", "entropy_growth", "resonance_coherence",
            "branch_factor", "attention_fragmentation", "recursion_depth", "token_acceleration"
        ]
        
        for item in raw_data:
            feat_dict = item["features"]
            # Extract features in fixed order
            vec = [feat_dict.get(k, 0.0) for k in self.feature_keys]
            self.X.append(vec)
            self.y.append(self.regime_to_idx.get(item["label"], 6)) # Default to mixed_mode
            
        self.X = torch.tensor(self.X, dtype=torch.float32)
        self.y = torch.tensor(self.y, dtype=torch.long)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def train_classifier(data_path: str, model_save_path: str = "models/regime_classifier.pt"):
    dataset = RegimeDataset(data_path)
    if len(dataset) < 10:
        print("Not enough data to train. Need at least 10 samples.")
        return
        
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    model = RegimeModel(input_dim=len(dataset.feature_keys), num_classes=len(dataset.regimes))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    model.train()
    for epoch in range(50):
        total_loss = 0
        correct = 0
        for x, y in loader:
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            correct += (out.argmax(1) == y).sum().item()
            
        acc = correct / len(dataset)
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {total_loss/len(loader):.4f}, Acc: {acc:.4f}")
            
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

if __name__ == "__main__":
    # Example usage
    # train_classifier("training/regime_data.json")
    pass
