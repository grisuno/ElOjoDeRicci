# API

## app.py
- `PTSymmetricActivation.__init__` (method) `app.py:32` `def __init__(self, omega, chi, kappa_init)`
- `PTSymmetricActivation.forward` (method) `app.py:37` `def forward(self, x)`
- `RicciCurvatureAttention.__init__` (method) `app.py:45` `def __init__(self, dim)`
- `RicciCurvatureAttention.forward` (method) `app.py:51` `def forward(self, x)`
- `E8LatticeLayer.__init__` (method) `app.py:60` `def __init__(self, in_f, out_f, edge_index, num_nodes)`
- `E8LatticeLayer.forward` (method) `app.py:70` `def forward(self, x)`
- `RESMAGraph.__init__` (method) `app.py:77` `def __init__(self, input_dim, hidden_dim, edge_index, num_nodes)`
- `RESMAGraph.forward` (method) `app.py:86` `def forward(self, x)`
- `RESMAGraph.load_elliptic_data` (method) `app.py:97` `def load_elliptic_data()`
- `MLP.__init__` (method) `app.py:158` `def __init__(self, input_dim, hidden_dim)`
- `MLP.forward` (method) `app.py:172` `def forward(self, x)`
- `SimpleGCN.__init__` (method) `app.py:177` `def __init__(self, input_dim, hidden_dim)`
- `SimpleGCN.forward` (method) `app.py:182` `def forward(self, x, edge_index)`
- `SimpleGCN.train_model` (method) `app.py:188` `def train_model(model, X, y, edge_index, name, epochs, lr)`
