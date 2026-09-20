# Troubleshooting

## `ModuleNotFoundError`

Activate the project environment and install the requirements:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

If the missing module begins with `step`, confirm that all nine files are in the
same folder and that every import matches the actual filename exactly.

## `SyntaxError` after adding number prefixes

Python module names used in standard imports cannot begin with a number. Use:

```text
step01_config.py
```

instead of:

```text
01_config.py
```

## Data folder not found

Use a raw Windows path string in `step01_config.py`:

```python
DATA_FOLDER = r"C:\Users\name\path\to\JSON_file"
```

## No usable data in a split

Check the earlier console messages. Common causes are:

- missing required JSON fields;
- fewer than three valid early reference capacities;
- charging curves with fewer than 20 valid voltage points;
- curves that do not cover 3.10–3.55 V;
- SOH values outside the configured limits;
- a file split containing no battery with usable cycles.

Adjust thresholds only after checking the raw measurements and explaining the
choice in the thesis methodology.

## TensorFlow installation problems

Confirm that the selected Visual Studio Code interpreter is the same environment
where TensorFlow was installed:

```powershell
python -c "import sys; print(sys.executable)"
python -c "import tensorflow as tf; print(tf.__version__)"
```

## Model training is slow

Confirm whether TensorFlow can detect an accelerator:

```powershell
python -c "import tensorflow as tf; print(tf.config.list_physical_devices())"
```

CPU training is valid but can take longer. Reduce `EPOCHS` temporarily when
checking that the pipeline works, then restore the intended experimental value.

## Poor test performance

Inspect file-level metrics and cycle plots. Possible causes include different
operating conditions across batteries, a small number of training batteries,
sensor noise, an unsuitable common voltage interval, or distribution differences
between splits. Do not tune the model using the final test results; make choices
with training and validation data, then evaluate the selected model once on the
test set.

