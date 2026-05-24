import torch
from transformers import SwinForImageClassification, SwinConfig

# Load checkpoint
checkpoint = torch.load(
    "swin_finetuned/best_weights.pt",
    map_location="cpu"
)

# Convert tensors to FP16
if isinstance(checkpoint, dict):

    if "model_state_dict" in checkpoint:
        checkpoint["model_state_dict"] = {
            k: v.half() if torch.is_tensor(v) else v
            for k, v in checkpoint["model_state_dict"].items()
        }

    elif "state_dict" in checkpoint:
        checkpoint["state_dict"] = {
            k: v.half() if torch.is_tensor(v) else v
            for k, v in checkpoint["state_dict"].items()
        }

    else:
        checkpoint = {
            k: v.half() if torch.is_tensor(v) else v
            for k, v in checkpoint.items()
        }

# Save compressed model
torch.save(
    checkpoint,
    "swin_finetuned/best_weights_fp16.pt"
)

print("Done")