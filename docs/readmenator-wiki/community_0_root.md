# root

*Community 0 | 2 files | cohesion 1.00*

## Definition

This community groups 2 file(s) rooted at `root` with dominant language py (cohesion 1.00). Central symbols: `E8LatticeLayer`, `MLP`, `PTSymmetricActivation`, `RESMAGraph`, `RicciCurvatureAttention`, `SimpleGCN`, `__init__`, `forward`. Core file: `app.py` (20 symbols). Documented purpose: Autor: Gris Iscomeback Correo electrónico: grisiscomeback[at]gmail[dot]com Fecha de creación: 24/12/2025 Licencia: GPL v3  Descripción: Regalo de Navidad..

## Files

| File | Language | Layer | Symbols | Doc |
|------|----------|-------|---------|-----|
| `app.py` | py | utility | 20 | yes |
| `install.sh` | sh | utility | 0 | no |

## Key Symbols

- `PTSymmetricActivation` (class, `app.py:31`) `class PTSymmetricActivation(Module)`
- `__init__` (method, `app.py:32`) `def __init__(self, omega, chi, kappa_init)`
- `forward` (method, `app.py:37`) `def forward(self, x)`
- `RicciCurvatureAttention` (class, `app.py:44`) `class RicciCurvatureAttention(Module)`
- `__init__` (method, `app.py:45`) `def __init__(self, dim)`
- `forward` (method, `app.py:51`) `def forward(self, x)`
- `E8LatticeLayer` (class, `app.py:59`) `class E8LatticeLayer(Module)`
- `__init__` (method, `app.py:60`) `def __init__(self, in_f, out_f, edge_index, num_nodes)`
- `forward` (method, `app.py:70`) `def forward(self, x)`
- `RESMAGraph` (class, `app.py:76`) `class RESMAGraph(Module)`
- `__init__` (method, `app.py:77`) `def __init__(self, input_dim, hidden_dim, edge_index, num_nodes)`
- `forward` (method, `app.py:86`) `def forward(self, x)`
- `load_elliptic_data` (method, `app.py:97`) `def load_elliptic_data()`
- `MLP` (class, `app.py:157`) `class MLP(Module)`
- `__init__` (method, `app.py:158`) `def __init__(self, input_dim, hidden_dim)`
- `forward` (method, `app.py:172`) `def forward(self, x)`
- `SimpleGCN` (class, `app.py:175`) `class SimpleGCN(Module)`
- `__init__` (method, `app.py:177`) `def __init__(self, input_dim, hidden_dim)`
- `forward` (method, `app.py:182`) `def forward(self, x, edge_index)`
- `train_model` (method, `app.py:188`) `def train_model(model, X, y, edge_index, name, epochs, lr)`

## Internal vs External Edges

- Internal resolved imports (EXTRACTED): 0
- Cross-boundary resolved imports (EXTRACTED): 0

## Connections

- No cross-community bridges recorded. This community is self-contained.

## Risks

- No scoped security, taint, cycle, or layer risks.

## Open Questions

- Why do 1 file(s) lack file-level docs (e.g. `install.sh`)? What purpose do they serve?
- What would break if the most connected file in root changed?
- Should root be split, given cohesion 1.00?

## Sources

- `app.py`
- `install.sh`
