# Installation and required libraries

## Recommended environment

Use a separate virtual environment so the project libraries do not interfere
with other Python projects. A recent 64-bit Python version supported by your
TensorFlow release is required.

From the project folder on Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

In Visual Studio Code, select the new environment with **Python: Select
Interpreter** from the Command Palette.

## Libraries

| Library | Purpose in this project |
|---|---|
| NumPy | Arrays, interpolation grids, gradients and numerical calculations |
| Pandas | Data tables, missing-value interpolation and CSV output |
| Matplotlib | Training and prediction plots |
| scikit-learn | Standardization and regression metrics |
| TensorFlow/Keras | CNN construction, training, checkpointing and prediction |

Python's built-in `os`, `json` and `random` modules require no separate
installation.

## Verify the installation

```powershell
python -c "import numpy, pandas, matplotlib, sklearn, tensorflow; print('Libraries are available')"
```

## Reproducible library versions

After obtaining final thesis results, save the exact installed versions:

```powershell
python -m pip freeze > requirements-lock.txt
```

Keep `requirements.txt` as the short human-readable dependency list and commit
`requirements-lock.txt` with the experiment used for the final reported result.

