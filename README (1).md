# Cervical Cancer Fusion Model — CI/CD Deployment

## Files

| File | Purpose |
|---|---|
| `train_meta_learner.py` | Retrain meta-learner on paired patients, log to MLflow, promote if better |
| `app.py` | Gradio UI — loads Production model from MLflow automatically |
| `run_pipeline.py` | Full CI/CD pipeline — retrain + gate + restart Gradio |
| `requirements.txt` | Python dependencies |

## Folder structure expected

```
project/
├── tabpfn33_model.pkl
├── swin_finetuned/
│   └── best_weights.pt
├── meta_learner_deployment/
│   ├── meta_learner.pkl        ← saved by train_meta_learner.py
│   └── meta_config.pkl         ← config including NORMAL_IDXS, AUC etc.
├── train_meta_learner.py
├── app.py
└── run_pipeline.py
```

## Setup

```bash
pip install -r requirements.txt

# Start MLflow tracking server (run in a separate terminal)
mlflow server --host 0.0.0.0 --port 5000
```

## First run

```bash
# 1. Edit PAIRED_CSV path in train_meta_learner.py
# 2. Run full pipeline
python run_pipeline.py
```

## Daily CI/CD — when new patients arrive

```bash
# Append new patients to paired_patients.csv
# Then run the pipeline — it auto-promotes only if AUC improves
python run_pipeline.py
```

## Check what's deployed

```bash
python run_pipeline.py --check-registry
```

## How the CI/CD gate works

```
New data arrives
      ↓
train_meta_learner.py runs
      ↓
MLflow logs: AUC, F1, params, model artifact
      ↓
Gate: new AUC >= current Production AUC?
      ├── YES → promote to Production → restart Gradio
      └── NO  → archive new version → keep current Production
```

## Paired patients CSV format

```csv
Age,Number of sexual partners,...,Biopsy,image_path
25,2,...,0,/path/to/patient1.png
31,1,...,1,/path/to/patient2.png
```

- All tabular feature columns must be in the same order as during original TabPFN training
- `Biopsy`: binary label (0=Normal, 1=Abnormal)  
- `image_path`: absolute path to the cytology image for that patient

## Gradio app

```bash
python app.py
# Open http://localhost:7860
```

The app always loads the **Production** stage model from MLflow automatically.
No code changes needed when a new model is promoted.
