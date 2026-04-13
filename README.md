# Dyslexia Handwriting MVP

This is a buildable MVP for handwriting-based dyslexia screening.

Input: handwriting image  
Output: `dyslexic` / `non_dyslexic` + confidence  
Serving: Flask API and simple web UI

## 1) Setup

Install dependencies in your Python 3.10 venv:

```bash
pip install -r requirements.txt
```

## 2) Download Dataset

Run:

```bash
python scripts/download_dataset.py
```

If archives are password-protected, pass password directly:

```bash
python scripts/download_dataset.py --password WanAsy321
```

Or use environment variable `DATASET_PASSWORD`.

This dataset uses `Gambo/Train|Test/Corrected|Normal|Reversal`.
Default binary mapping is:

- `Normal` -> `non_dyslexic`
- `Reversal` -> `dyslexic`
- `Corrected` -> `dyslexic` (configurable)

Recommended import command (fresh rebuild):

```bash
python scripts/download_dataset.py --password WanAsy321 --reset-target --corrected-as dyslexic
```

This downloads the Kaggle dataset and tries to copy images into:

```text
dataset/
  dyslexic/
  non_dyslexic/
```

If class folders are not auto-detected, manually copy files into those two folders.

## 3) Train Model

```bash
python train.py --data-dir dataset --epochs 15 --batch-size 16 --lr 1e-4
```

Model checkpoint is saved to `artifacts/convnext_mvp.pt`.

## 4) Run API + UI

```bash
python app.py
```

Open `http://127.0.0.1:5000`.

## 5) API Contract

Endpoint:

```http
POST /predict
Content-Type: multipart/form-data
```

Body field: `image`

Response:

```json
{
  "prediction": "dyslexic",
  "confidence": 0.87,
  "label": "Needs further evaluation"
}
```

`label` is returned when confidence is below `0.70`.

## Notes

- This is an educational support tool, not a medical diagnosis.
- Start with ConvNeXt-Tiny MVP. Add Swin fusion in Phase 2.
