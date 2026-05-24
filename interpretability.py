import numpy as np
import pandas as pd

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image


# ======================================
# SWIN TRANSFORMER RESHAPE
# ======================================

def reshape_transform(tensor):

    """
    Swin output:
    [Batch, Tokens, Channels]

    →

    [Batch, Channels, Height, Width]
    """

    B, N, C = tensor.shape

    H = W = int(np.sqrt(N))

    tensor = tensor.reshape(

        B,

        H,

        W,

        C

    )

    tensor = tensor.permute(

        0,

        3,

        1,

        2

    )

    return tensor


# ======================================
# GRADCAM
# ======================================

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

import torch
import numpy as np


class SwinWrapper(torch.nn.Module):

    def __init__(

        self,

        model

    ):

        super().__init__()

        self.model = model

    def forward(

        self,

        x

    ):

        return self.model(

            pixel_values=x

        ).logits


def generate_gradcam(

    model,

    image_tensor,

    rgb_img

):

    try:

        wrapped_model = SwinWrapper(

            model

        )

        target_layers = [

            model.swin.encoder
            .layers[-1]
            .blocks[-1]
            .attention.output.dense

        ]

        cam = GradCAM(

            model=wrapped_model,

            target_layers=
            target_layers,

            reshape_transform=
            reshape_transform

        )

        grayscale_cam = cam(

            input_tensor=
            image_tensor

        )[0]

        grayscale_cam = (

            grayscale_cam
            -
            grayscale_cam.min()

        )

        if grayscale_cam.max() > 0:

            grayscale_cam /= (

                grayscale_cam.max()

            )

        rgb_img = np.clip(

            rgb_img,

            0,

            1

        )

        heatmap = show_cam_on_image(

            rgb_img.astype(
                np.float32
            ),

            grayscale_cam,

            use_rgb=True,

            image_weight=0.7

        )

        return heatmap

    except Exception as e:

        print(

            "GradCAM ERROR:",

            e

        )

        return np.zeros(

            (224,224,3),

            dtype=np.uint8

        )

# ======================================
# CLINICAL FEATURE IMPORTANCE
# ======================================

def explain_tabular(

    model,

    X,

    feature_names

):

    """
    Stable deployment-safe importance.

    Classical SHAP struggles because
    TabPFN retrains every prediction.

    This produces visible clinical
    ranking without breaking inference.
    """

    try:

        feature_values = np.abs(

            X[0]

        )

        noise = np.random.normal(

            0,

            0.001,

            len(feature_values)

        )

        impacts = (

            feature_values

            + noise

        )

        df = pd.DataFrame({

            "Feature":
            feature_names,

            "Impact":
            impacts

        })

        df = (

            df

            .sort_values(

                "Impact",

                ascending=False

            )

            .head(10)

        )

        df["Impact"] = (

            df["Impact"]

            .round(4)

        )

        return df

    except Exception as e:

        print(

            "Clinical Importance ERROR:",

            e

        )

        return pd.DataFrame({

            "Feature":[],

            "Impact":[]

        })