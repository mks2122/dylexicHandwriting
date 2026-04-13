from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.models import ConvNeXt_Tiny_Weights, convnext_tiny


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_model(pretrained: bool = True):
    if pretrained:
        weights = ConvNeXt_Tiny_Weights.DEFAULT
        model = convnext_tiny(weights=weights)
    else:
        model = convnext_tiny(weights=None)

    in_features = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(in_features, 1)
    return model


def get_eval_transform():
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


class Predictor:
    def __init__(self, checkpoint_path: str, threshold: float = 0.5):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.threshold = threshold
        self.transform = get_eval_transform()

        checkpoint_file = Path(checkpoint_path)
        if not checkpoint_file.exists():
            raise FileNotFoundError(
                f"Checkpoint not found at {checkpoint_file.resolve()}"
            )

        # PyTorch 2.6+ defaults to weights_only=True which can fail for full training checkpoints.
        checkpoint = torch.load(
            checkpoint_file,
            map_location=self.device,
            weights_only=False,
        )
        self.class_to_idx = checkpoint.get(
            "class_to_idx", {"dyslexic": 0, "non_dyslexic": 1}
        )

        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        self.model = build_model(pretrained=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def predict(self, image: Image.Image):
        tensor = self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)
        logits = self.model(tensor)
        prob_positive = torch.sigmoid(logits).item()

        # Positive class is mapped as class index 1 in the training target.
        pred_idx = 1 if prob_positive >= self.threshold else 0
        pred_label = self.idx_to_class.get(pred_idx, "non_dyslexic")

        confidence = prob_positive if pred_idx == 1 else (1.0 - prob_positive)

        return {
            "prediction": pred_label,
            "confidence": round(float(confidence), 4),
            "raw_probability": round(float(prob_positive), 4),
        }
