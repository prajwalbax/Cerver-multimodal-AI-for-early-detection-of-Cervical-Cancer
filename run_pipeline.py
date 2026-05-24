# ================== CRITICAL FIX (TOP OF FILE) ==================
import torch

# Force ALL torch loads to CPU (fixes your crash)
torch_load_original = torch.load
def torch_load_cpu(*args, **kwargs):
    kwargs['map_location'] = torch.device('cpu')
    return torch_load_original(*args, **kwargs)

torch.load = torch_load_cpu
# ===============================================================


import os
import math
import json
import joblib
import warnings
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from datetime import datetime
from pathlib import Path
from PIL import Image

from torchvision import transforms
from transformers import SwinForImageClassification, SwinConfig

from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    classification_report, roc_auc_score,
    f1_score, precision_score, recall_score, confusion_matrix
)

warnings.filterwarnings("ignore")

# ================== PATHS (FIXED) ==================
TAB_MODEL_PATH  = "tabpfn33_model.pkl"

IMG_MODEL_PATH  = Path(
    r"C:/Users/praj/OneDrive/Desktop/multimodal cerver deployment/swin_finetuned/best_weights.pt"
)

META_CONFIG_PATH = "meta_learner_deployment/meta_config.pkl"

PAIRED_CSV = Path(
    r"C:\Users\praj\OneDrive\Desktop\multimodal cerver deployment\updated_images_final (1).csv"
)

LABEL_COL       = "Biopsy"
IMAGE_PATH_COL  = "Image_path"

TEST_SIZE    = 0.29
RANDOM_STATE = 42

MLFLOW_EXPERIMENT = "cervical_cancer_fusion"
MODEL_NAME        = "meta_learner_fusion"

DEPLOY_DIR = "meta_learner_deployment"
os.makedirs(DEPLOY_DIR, exist_ok=True)

DEVICE = torch.device("cpu")   # FORCE CPU (stable deployment)
# ==================================================


# ================== LOAD MODELS ==================
from tabpfn import TabPFNClassifier

def load_tabpfn(path=None):
    print("Loading TabPFN fresh (recommended)...")
    model = TabPFNClassifier(device='cpu')
    return model


def load_swin(path, num_classes=5):
    checkpoint = torch.load(path)

    state_dict = (
        checkpoint.get("model_state_dict",
        checkpoint.get("state_dict", checkpoint))
        if isinstance(checkpoint, dict) else None
    )

    if state_dict is None:
        return checkpoint.to(DEVICE).eval(), 224

    bias_key  = next(k for k in state_dict if "relative_position_bias_table" in k)
    bias_rows = state_dict[bias_key].shape[0]
    win_size  = int((math.sqrt(bias_rows) + 1) / 2)
    img_size  = {7: 224, 8: 256, 4: 128}.get(win_size, 224)

    cfg = SwinConfig(
        image_size=img_size,
        num_labels=num_classes,
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=win_size,
        patch_size=4,
    )

    model = SwinForImageClassification(cfg)
    model.load_state_dict(state_dict, strict=True)
    model.to(DEVICE).eval()

    print(f"Swin loaded: {img_size}x{img_size}")
    return model, img_size
# ==================================================


def get_img_transform(img_size):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std =[0.229, 0.224, 0.225]
        ),
    ])


def get_tabpfn_probs(tab_model, X_train, y_train, X_test):
    # Fit on training data
    tab_model.fit(X_train, y_train)

    # Predict probabilities
    return tab_model.predict_proba(X_test).astype(np.float32)


def get_swin_probs(swin_model, image_paths, transform, batch_size=16):
    all_probs = []

    with torch.no_grad():
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i+batch_size]
            tensors = []

            for p in batch_paths:
                img = Image.open(p).convert("RGB")
                tensors.append(transform(img))

            batch = torch.stack(tensors).to(DEVICE)
            logits = swin_model(pixel_values=batch).logits
            probs  = torch.softmax(logits, dim=1).cpu().numpy()

            all_probs.append(probs)

    return np.concatenate(all_probs, axis=0).astype(np.float32)


def build_meta_features(p_tab, p_img, normal_idxs):
    X_meta = np.concatenate([p_tab, p_img], axis=1)

    p_img_normal = p_img[:, normal_idxs].sum(axis=1)
    p_img_2class = np.stack([p_img_normal, 1 - p_img_normal], axis=1)

    return X_meta, p_img_2class


# ================== MAIN ==================
def main():
    print("=" * 60)
    print("META-LEARNER RETRAINING")
    print("=" * 60)

    tab_model = load_tabpfn(TAB_MODEL_PATH)
    swin_model, IMG_SZ = load_swin(IMG_MODEL_PATH)

    transform = get_img_transform(IMG_SZ)

    cfg = joblib.load(META_CONFIG_PATH)
    normal_idxs = cfg["normal_idxs"]

    df = pd.read_csv(PAIRED_CSV)

    # Keep only numeric columns automatically
    numeric_df = df.select_dtypes(include=[np.number])

    # Remove label column from features
    X_tab = numeric_df.drop(columns=[LABEL_COL]).values.astype(np.float32)

    # Labels
    y = numeric_df[LABEL_COL].values.astype(int)

    # Image paths (keep separately)
    paths = df[IMAGE_PATH_COL].tolist()

    X_tr, X_te, y_tr, y_te, p_tr, p_te = train_test_split(
        X_tab, y, paths,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE
    )

    print("Running TabPFN...")
    

    # Train once
    tab_model.fit(X_tr, y_tr)

    # Predict
    p_tab_tr = tab_model.predict_proba(X_tr).astype(np.float32)
    p_tab_te = tab_model.predict_proba(X_te).astype(np.float32)
    print("Running Swin...")
    p_img_tr = get_swin_probs(swin_model, p_tr, transform)
    p_img_te = get_swin_probs(swin_model, p_te, transform)

    X_meta_tr, _ = build_meta_features(p_tab_tr, p_img_tr, normal_idxs)
    X_meta_te, _ = build_meta_features(p_tab_te, p_img_te, normal_idxs)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_meta_tr, y_tr)

    y_pred = model.predict(X_meta_te)
    auc    = roc_auc_score(y_te, model.predict_proba(X_meta_te)[:,1])



    joblib.dump(model, os.path.join(DEPLOY_DIR, "meta_learner.pkl"))
    print("Saved model ✔")


if __name__ == "__main__":
    main()