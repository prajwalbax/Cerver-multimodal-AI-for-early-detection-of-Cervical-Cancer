import os
import math
import warnings
import numpy as np
import pandas as pd
import joblib
import gradio as gr
import torch

from torchvision import transforms
from transformers import (
    SwinForImageClassification,
    SwinConfig
)

from PIL import Image
from tabpfn import TabPFNClassifier

from interpretability import (
    generate_gradcam,
    explain_tabular
)

warnings.filterwarnings("ignore")


# ================= CONFIG =================

IMG_MODEL_PATH = (
    "swin_finetuned/"
    "best_weights_fp16.pt"
)

META_CONFIG_PATH = (
    "meta_learner_deployment/"
    "meta_config.pkl"
)

META_LOCAL_PATH = (
    "meta_learner_deployment/"
    "meta_learner.pkl"
)

DEVICE = torch.device("cpu")


FEATURE_NAMES = [

    "Age",
    "Number of sexual partners",
    "First sexual intercourse",
    "Num of pregnancies",
    "Smokes",
    "Smokes (years)",
    "Smokes (packs/year)",
    "Hormonal Contraceptives",
    "Hormonal Contraceptives (years)",
    "IUD",
    "IUD (years)",
    "STDs",
    "STDs (number)",
    "STDs:condylomatosis",
    "STDs:cervical condylomatosis",
    "STDs:vaginal condylomatosis",
    "STDs:vulvo-perineal condylomatosis",
    "STDs:syphilis",
    "STDs:pelvic inflammatory disease",
    "STDs:genital herpes",
    "STDs:molluscum contagiosum",
    "STDs:AIDS",
    "STDs:HIV",
    "STDs:Hepatitis B",
    "STDs:HPV",
    "STDs: Number of diagnosis",
    "STDs: Time since first diagnosis",
    "STDs: Time since last diagnosis",
    "Dx:Cancer",
    "Dx:CIN",
    "Dx:HPV",
    "Dx",
    "Hinselmann",
    "Schiller",
    "Citology"

]


CYTOLOGY_CLASSES = [

    "Dyskeratotic (HSIL/SCC)",

    "Koilocytotic (LSIL)",

    "Metaplastic (ASCUS II)",

    "Parabasal (ASCUS I)",

    "Superficial_Intermediate (Normal)"

]


# ================= MODEL LOAD =================

def load_all_models():

    print("Loading models ...")

    tab_model = TabPFNClassifier(
        device="cpu"
    )

    print(
        "  TabPFN : initialized"
    )

    checkpoint = torch.load(
        IMG_MODEL_PATH,
        map_location=DEVICE
    )

    state_dict = (

        checkpoint.get(
            "model_state_dict",

            checkpoint.get(
                "state_dict",
                checkpoint
            )

        )

        if isinstance(
            checkpoint,
            dict
        )

        else checkpoint

    )

    bias_key = next(

        k

        for k in state_dict

        if "relative_position_bias_table"
        in k

    )

    bias_rows = (
        state_dict[
            bias_key
        ].shape[0]
    )

    win_size = int(

        (
            math.sqrt(
                bias_rows
            ) + 1
        ) / 2

    )

    img_size = {

        7:224,

        8:256,

        4:128

    }.get(
        win_size,
        224
    )

    cfg = SwinConfig(

        image_size=img_size,

        num_labels=5,

        embed_dim=96,

        depths=[2,2,6,2],

        num_heads=[3,6,12,24],

        window_size=win_size,

        patch_size=4

    )

    swin = SwinForImageClassification(
        cfg
    )

    swin.load_state_dict(
        state_dict
    )

    swin.float()

    swin.eval()

    swin.to(DEVICE)

    transform = transforms.Compose([

        transforms.Resize(
            (img_size,img_size)
        ),

        transforms.ToTensor(),

        transforms.Normalize(

            mean=[
                0.485,
                0.456,
                0.406
            ],

            std=[
                0.229,
                0.224,
                0.225
            ]

        )

    ])

    meta_model = joblib.load(
        META_LOCAL_PATH
    )

    meta_cfg = joblib.load(
        META_CONFIG_PATH
    )

    return (

        tab_model,

        swin,

        transform,

        meta_model,

        meta_cfg

    )


tab_model,\
swin_model,\
img_transform,\
meta_model,\
meta_cfg = load_all_models()


CONF_THRESHOLD = (
    meta_cfg.get(
        "confidence_threshold",
        0.75
    )
)


# ================= PREDICT =================

def predict(

    image,

    *tab_values

):

    try:

        if image is None:

            return (

                "Upload image",

                None,

                None,

                None,

                pd.DataFrame()

            )

        X = np.array(

            [[

                float(v)

                for v

                in tab_values

            ]],

            dtype=np.float32

        )

        noise = np.random.normal(

            0,

            1e-3,

            X.shape

        )

        X_safe = np.vstack([

            X,

            X + noise

        ])

        tab_model.fit(

            X_safe,

            np.array([0,1])

        )

        p_tab = (

            tab_model
            .predict_proba(
                X_safe
            )[:1]

        )

        img = (

            image

            if isinstance(
                image,
                Image.Image
            )

            else Image.fromarray(
                image
            )

        )

        img = img.convert("RGB")

        img_tensor = (

            img_transform(img)

            .unsqueeze(0)

            .to(DEVICE)

        )

        with torch.no_grad():

            logits = swin_model(

                pixel_values=
                img_tensor

            ).logits

        p_img = (

            torch.softmax(

                logits,

                dim=1

            )

            .cpu()

            .numpy()

        )

        X_meta = np.concatenate(

            [

                p_tab,

                p_img

            ],

            axis=1

        )

        final = (

            meta_model
            .predict_proba(
                X_meta
            )

        )

        risk = float(
            final[0,1]
        )

        conf = float(
            np.max(final)
        )

        label = (

            "ABNORMAL"

            if risk>=0.5

            else "NORMAL"

        )

        gradcam = generate_gradcam(

            swin_model,

            img_tensor,

            np.array(img)/255.0

        )

        shap_df = explain_tabular(

            tab_model,

            X,

            FEATURE_NAMES

        )

        return (

            f"Prediction: {label}\n"
            f"Risk: {risk:.3f}\n"
            f"Confidence:{conf:.3f}",

            pd.DataFrame({

                "Class":
                CYTOLOGY_CLASSES,

                "Probability":
                p_img[0]

            }),

            f"Threshold:{CONF_THRESHOLD}",

            gradcam,

            shap_df

        )

    except Exception as e:

        print(e)

        return (

            f"Error: {e}",

            pd.DataFrame(),

            "Failed",

            None,

            pd.DataFrame()

        )


# ================= UI =================

def build_app():

    with gr.Blocks() as demo:

        gr.Markdown(
            "# Cervical Cancer Detection"
        )

        with gr.Row():

            with gr.Column():

                img = gr.Image(
                    type="pil"
                )

                inputs=[

                    gr.Number(
                        label=f,
                        value=0
                    )

                    for f

                    in FEATURE_NAMES

                ]

                btn=gr.Button(
                    "Predict"
                )

            with gr.Column():

                result=gr.Textbox()

                probs=gr.Dataframe()

                info=gr.Textbox()

                gradcam=gr.Image(

                    label="GradCAM"

                )

                shap=gr.Dataframe(

                    label="SHAP"

                )

        btn.click(

            predict,

            inputs=[img]+inputs,

            outputs=[

                result,

                probs,

                info,

                gradcam,

                shap

            ]

        )

    return demo


if __name__=="__main__":

    app=build_app()

    app.launch(

        server_name="0.0.0.0",

        server_port=7861

    )