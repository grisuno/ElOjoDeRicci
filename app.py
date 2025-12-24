#!/usr/bin/env python3
# _*_ coding: utf8 _*_
"""
app.py

Autor: Gris Iscomeback
Correo electrónico: grisiscomeback[at]gmail[dot]com
Fecha de creación: 24/12/2025
Licencia: GPL v3

Descripción: Regalo de Navidad.
"""
import os
import glob
import torch
import zipfile
import kagglehub
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch_geometric.utils import add_self_loops
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score, f1_score



class PTSymmetricActivation(nn.Module):
    def __init__(self, omega=1.0, chi=0.5, kappa_init=1.0):
        super().__init__()
        self.kappa = nn.Parameter(torch.tensor(float(kappa_init)))
        self.threshold = chi * omega
        
    def forward(self, x):
        x_safe = torch.clamp(x, -10.0, 10.0)
        coherence = self.kappa / (self.threshold + 1e-8)
        zeeman = torch.pow(torch.abs(x_safe), 4.0) * 1e-2  
        gate = torch.sigmoid(coherence - zeeman)
        return x * gate

class RicciCurvatureAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.tau = nn.Parameter(torch.tensor(0.5))
        
    def forward(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        q, k = self.q(x), self.k(x)
        metric = torch.matmul(q, k.transpose(-2, -1)) * self.tau
        attn = torch.softmax(metric, dim=-1)
        return torch.matmul(attn, x).squeeze(1)

class E8LatticeLayer(nn.Module):
    def __init__(self, in_f, out_f, edge_index, num_nodes):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(out_f, in_f))
        nn.init.xavier_uniform_(self.weights)

        values = torch.ones(edge_index.shape[1])

        self.register_buffer('topology_sparse', torch.sparse_coo_tensor(edge_index, values, (num_nodes, num_nodes)))
        self.num_nodes = num_nodes
        
    def forward(self, x):
        neighbor_agg = torch.sparse.mm(self.topology_sparse, x)
        
        out = F.linear(neighbor_agg, self.weights)
        return out

class RESMAGraph(nn.Module):
    def __init__(self, input_dim, hidden_dim, edge_index, num_nodes):
        super().__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.e8 = E8LatticeLayer(hidden_dim, hidden_dim, edge_index, num_nodes)
        self.ricci = RicciCurvatureAttention(hidden_dim)
        self.pt_act = PTSymmetricActivation()
        self.norm = nn.LayerNorm(hidden_dim)
        self.readout = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        x = self.embedding(x)
        for _ in range(2):
            res = x
            x = self.e8(x)
            x = self.ricci(x)
            x = self.pt_act(x)
            x = self.norm(x + res)
        return torch.sigmoid(self.readout(x))


def load_elliptic_data():
    print("Descargando Elliptic Data Set...")
    path = kagglehub.dataset_download("ellipticco/elliptic-data-set")

    zip_files = glob.glob(os.path.join(path, "*.zip"))
    if zip_files:
        zip_path = zip_files[0]
        print(f"Descomprimiendo {zip_path} en {path}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(path)

    csv_files = glob.glob(os.path.join(path, "**", "elliptic_txs_features.csv"), recursive=True)
    if not csv_files:
        raise FileNotFoundError("No se encontró elliptic_txs_features.csv ni en raíz ni en subcarpetas.")
    
    base_dir = os.path.dirname(csv_files[0])  
    print(f"Archivos encontrados en: {base_dir}")
    
    features_path = os.path.join(base_dir, "elliptic_txs_features.csv")
    classes_path = os.path.join(base_dir, "elliptic_txs_classes.csv")
    edgelist_path = os.path.join(base_dir, "elliptic_txs_edgelist.csv")

    nodes = pd.read_csv(features_path, header=None)
    classes = pd.read_csv(classes_path)
    edges = pd.read_csv(edgelist_path)

    nodes.columns = ["txId"] + ["timestep"] + [f"feat_{i}" for i in range(1, 166)]
    df = nodes.merge(classes, on="txId", how="left")
    
    
    df = df[df["class"].isin(["1", "2"])]
    df["class"] = df["class"].map({"1": 1, "2": 0}).astype(int)
    
    X = df.iloc[:, 1:-1].values  
    y = df["class"].values
    
    
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    
    tx_to_idx = {tx: i for i, tx in enumerate(df["txId"])}
    edge_df = edges[edges["txId1"].isin(tx_to_idx) & edges["txId2"].isin(tx_to_idx)]
    edge_index = torch.tensor([
        [tx_to_idx[tx] for tx in edge_df["txId1"]],
        [tx_to_idx[tx] for tx in edge_df["txId2"]]
    ], dtype=torch.long)
    
    edge_index, _ = add_self_loops(edge_index, num_nodes=len(X))
    
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    
    print(f"Nodos etiquetados: {len(X)}")
    print(f"Ilícitos: {y.sum().item()} ({y.mean().item()*100:.2f}%)")
    print(f"Aristas: {edge_index.shape[1]}")
    
    return X, y, edge_index


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x)

class SimpleGCN(nn.Module):

    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.conv1 = self.GCNConv(input_dim, hidden_dim)
        self.conv2 = self.GCNConv(hidden_dim, hidden_dim)
        self.readout = nn.Linear(hidden_dim, 1)
    def forward(self, x, edge_index):
        x = F.leaky_relu(self.conv1(x, edge_index))
        x = F.leaky_relu(self.conv2(x, edge_index))
        return torch.sigmoid(self.readout(x))


def train_model(model, X, y, edge_index=None, name="Model", epochs=50, lr=0.001):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    best_auprc = 0
    patience = 0
    

    train_idx, val_idx = train_test_split(
        np.arange(len(X)), test_size=0.3, stratify=y.numpy(), random_state=42
    )
    train_idx = torch.tensor(train_idx)
    val_idx = torch.tensor(val_idx)
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        
        if hasattr(model, 'e8'):  
            out = model(X)  
        elif edge_index is not None:  
            out = model(X, edge_index)
        else:  
            out = model(X)
            
        
        loss = F.binary_cross_entropy(out[train_idx], y[train_idx])
        loss.backward()
        optimizer.step()
        
        
        model.eval()
        with torch.no_grad():
            if hasattr(model, 'e8'):
                val_out = model(X)[val_idx].cpu().numpy()
            elif edge_index is not None:
                val_out = model(X, edge_index)[val_idx].cpu().numpy()
            else:
                val_out = model(X[val_idx]).cpu().numpy()
                
            val_y = y[val_idx].cpu().numpy()
            auprc = average_precision_score(val_y, val_out)
            
        if auprc > best_auprc:
            best_auprc = auprc
            patience = 0
        else:
            patience += 1
            if patience > 10:
                break
                
        if epoch % 10 == 0:
            print(f"{name} | Epoch {epoch} | AUPRC: {auprc:.4f}")
            
    return best_auprc

if __name__ == "__main__":
    X, y, edge_index = load_elliptic_data()
    num_nodes = X.shape[0]

    models = []

    mlp = MLP(X.shape[1], 128)
    mlp_auprc = train_model(mlp, X, y, name="MLP")
    
    
    gcn = SimpleGCN(X.shape[1], 128)
    gcn_auprc = train_model(gcn, X, y, edge_index, name="GCN")
    
    
    resma = RESMAGraph(X.shape[1], 128, edge_index, num_nodes)
    resma_auprc = train_model(resma, X, y, edge_index=None, name="RESMA")  

    results = [("MLP", mlp_auprc), ("GCN", gcn_auprc), ("RESMA", resma_auprc)]
    results.sort(key=lambda x: x[1], reverse=True)
    
    print("\n" + "="*50)
    print("RESULTADOS (AUPRC)")
    print("="*50)
    for i, (name, auprc) in enumerate(results, 1):
        marker = "🏆 " if i == 1 else "   "
        print(f"{marker}{i}. {name:10} | AUPRC: {auprc:.4f}")
