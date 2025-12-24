# RESMA: Ricci-E8 Symmetric Multi-Agent Graph Networks for Illegal Transaction Detection 

<img width="1024" height="1024" alt="image" src="https://github.com/user-attachments/assets/ee2c833c-27e2-4788-86f6-1f96de180419" />


**Abstract:** Identifying illicit activities in decentralized blockchain networks remains a critical challenge due to the high-dimensional noise and the adversarial nature of transaction obfuscation. We present ARGOS-E8, a novel non-Euclidean Graph Neural Network (GNN) architecture that leverages the intrinsic geometric properties of transaction graphs. By integrating a message-passing scheme based on the Gosset $E_8$ Lattice, an attention mechanism driven by Discrete Ricci Curvature, and a Parity-Time (PT) Symmetric activation function, our model captures structural anomalies that escape traditional Euclidean methods. Evaluated on the Elliptic Bitcoin dataset, ARGOS-E8 achieves a state-of-the-art AUPRC of 0.8731, demonstrating superior robustness in detecting money laundering patterns.

## 1. Introduction

Traditional Graph Convolutional Networks (GCNs) operate under the assumption of a flat, Euclidean latent space. However, transactional networks often exhibit "small-world" properties and hyperbolic growth, where the connectivity patterns of illicit actors differ fundamentally from legitimate commerce. Criminal strategies, such as "layering" and "smurfing," create localized structural deformations. We propose that these irregularities are best identified through differential geometry. ARGOS-E8 treats the transaction graph not just as a set of points, but as a manifold whose curvature reveals the underlying intent of the actors.

## 2. Methodology: The ARGOS-E8 Architecture

### 2.1 Higher-Order Aggregation via $E_8$ Lattice

Standard GNNs aggregate information from immediate neighbors in a linear fashion. ARGOS-E8 utilizes a topology inspired by the $E_8$ Lattice—the densest sphere packing in eight dimensions. This allows the model to map node features into a high-dimensional symmetric space, capturing complex multi-agent correlations before projecting them back for classification. This prevents the "oversmoothing" problem common in deep GNNs.

### 2.2 Discrete Ricci Curvature Attention

We implement an attention mechanism based on Ollivier-Ricci Curvature. In financial networks, curvature serves as a proxy for information "congestion" or "diffusion":

- Positive Curvature: Indicates dense, stable clusters typical of regulated exchanges and merchant hubs.

- Negative Curvature: Highlights "bridge" nodes and star-like expansion patterns often associated with fund dispersion in money laundering. The attention weights are dynamically adjusted based on the curvature of the edges, forcing the model to scrutinize regions where the "geometry" of the money flow is distorted.

### 2.3 PT-Symmetric Activation Function

Inspired by non-Hermitian quantum mechanics, we introduce the PT-Symmetric Activation. Unlike standard ReLU or Sigmoid functions that can be blinded by high-volume "noise" (massive legitimate transactions), our activation function uses a coherence-gate. It amplifies signals that maintain structural symmetry across the temporal axis while suppressing stochastic noise, acting as a sophisticated filter for "clean" vs. "tainted" data flows.

## 3. Experimental Analysis and Results

The model was benchmarked against Multi-Layer Perceptrons (MLP) and standard Graph Convolutional Networks (GCN) using the Elliptic dataset (203,769 nodes representing Bitcoin transactions).

### 3.1 Performance Metrics

Given the class imbalance (only ~10% of nodes are illicit), we prioritize the Area Under the Precision-Recall Curve (AUPRC).

```text
==================================================
RESULTADOS (AUPRC)
==================================================
🏆 1. RESMA      | AUPRC: 0.8731
   2. GCN        | AUPRC: 0.7497
   3. MLP        | AUPRC: 0.6565
```
### 3.2 Discussion

The significant jump in AUPRC (+16.4% over GCN) suggests that the "exotic" components are not merely aesthetic. The Ricci Curvature attention successfully flagged "mixer" services that GCNs often misclassify as standard high-volume hubs. Furthermore, the PT-Symmetric activation proved essential in maintaining model stability across different time-steps of the blockchain.

## 4. Conclusion
ARGOS-E8 demonstrates that the future of financial surveillance lies in the marriage of Graph Deep Learning and Physics-Informed Geometry. By moving away from flat architectures, we can identify criminal behavior based on the "shape" of the transaction, rather than just the volume.


![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) ![Shell Script](https://img.shields.io/badge/shell_script-%23121011.svg?style=for-the-badge&logo=gnu-bash&logoColor=white) ![Flask](https://img.shields.io/badge/flask-%23000.svg?style=for-the-badge&logo=flask&logoColor=white) [![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

[![ko-fi](https://ko-fi.com/img/githubbutton_sm.svg)](https://ko-fi.com/Y8Y2Z73AV)
