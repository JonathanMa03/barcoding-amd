from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms

import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
# TODO after clinician labels arrive:
# Use labels/predictions to select meaningful Grad-CAM examples:
# true positives, false positives, true negatives, false negatives.
# ---------------------------------------------------------------------


def make_gradcam_dirs(processed_dir: str | Path) -> dict[str, Path]:
    processed_dir = Path(processed_dir)

    dirs = {
        "processed": processed_dir,
        "figures": processed_dir / "figures" / "gradcam",
        "predictions": processed_dir / "predictions",
        "models": processed_dir / "models",
    }

    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    return dirs


class GradCAM:
    """
    Basic Grad-CAM implementation for ResNet-style models.
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer

        self.activations = None
        self.gradients = None

        self.forward_hook = target_layer.register_forward_hook(self._save_activation)
        self.backward_hook = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def remove_hooks(self):
        self.forward_hook.remove()
        self.backward_hook.remove()

    def generate(
        self,
        image_tensor: torch.Tensor,
        class_index: int = 1,
    ) -> np.ndarray:
        """
        Generate Grad-CAM heatmap.

        Parameters
        ----------
        image_tensor:
            Tensor with shape (1, 3, H, W).
        class_index:
            Target class. For binary barcode model, 1 = barcode-positive.

        Returns
        -------
        heatmap:
            Array with shape (H, W), normalized to [0, 1].
        """
        self.model.eval()
        self.model.zero_grad()

        logits = self.model(image_tensor)
        score = logits[:, class_index].sum()
        score.backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)

        cam = F.interpolate(
            cam,
            size=image_tensor.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        heatmap = cam.squeeze().cpu().numpy()
        heatmap = heatmap - heatmap.min()
        heatmap = heatmap / (heatmap.max() + 1e-8)

        return heatmap


def preprocess_bscan_for_resnet(
    bscan: np.ndarray,
    image_size: int = 224,
) -> torch.Tensor:
    """
    Convert one ROI B-scan to normalized ResNet tensor.
    """
    bscan = bscan.astype(np.float32)
    bscan = np.nan_to_num(bscan)

    bmin = bscan.min()
    bmax = bscan.max()
    bscan = (bscan - bmin) / (bmax - bmin + 1e-8)
    bscan = (255 * bscan).astype(np.uint8)

    img = Image.fromarray(bscan).convert("RGB")

    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    return transform(img).unsqueeze(0)


def bscan_to_display_image(
    bscan: np.ndarray,
    image_size: int = 224,
) -> np.ndarray:
    """
    Convert B-scan to displayable grayscale image in [0, 1].
    """
    bscan = bscan.astype(np.float32)
    bscan = np.nan_to_num(bscan)

    bmin = bscan.min()
    bmax = bscan.max()
    bscan = (bscan - bmin) / (bmax - bmin + 1e-8)

    img = Image.fromarray((255 * bscan).astype(np.uint8))
    img = img.resize((image_size, image_size))

    return np.asarray(img).astype(float) / 255.0


def overlay_heatmap(
    image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.45,
) -> np.ndarray:
    """
    Overlay heatmap on grayscale image.

    Returns RGB image in [0, 1].
    """
    cmap = plt.get_cmap("jet")
    heat_rgb = cmap(heatmap)[..., :3]

    image_rgb = np.stack([image, image, image], axis=-1)

    overlay = (1 - alpha) * image_rgb + alpha * heat_rgb
    overlay = np.clip(overlay, 0, 1)

    return overlay


def load_roi_volume(roi_path: str | Path) -> np.ndarray:
    roi_path = Path(roi_path)

    if not roi_path.exists():
        raise FileNotFoundError(f"ROI volume not found: {roi_path}")

    return np.load(roi_path)


def load_barcode_model(
    model_path: str | Path,
    build_model_fn,
    device: str | None = None,
):
    """
    Load trained barcode ResNet model.

    Parameters
    ----------
    model_path:
        Path to barcode_resnet.pt.
    build_model_fn:
        Function that returns model architecture, usually
        barcode.resnet.build_resnet50_binary.

    TODO after model training:
    Confirm checkpoint key structure if save format changes.
    """
    model_path = Path(model_path)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = build_model_fn(c8_checkpoint_path=None, freeze_backbone=False)

    checkpoint = torch.load(model_path, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()

    return model


def save_gradcam_figure(
    bscan: np.ndarray,
    heatmap: np.ndarray,
    output_path: str | Path,
    title: str | None = None,
    alpha: float = 0.45,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    display_img = bscan_to_display_image(bscan, image_size=heatmap.shape[0])
    overlay = overlay_heatmap(display_img, heatmap, alpha=alpha)

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(display_img, cmap="gray")
    plt.title("ROI B-scan")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(heatmap, cmap="jet")
    plt.title("Grad-CAM")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(overlay)
    plt.title("Overlay")
    plt.axis("off")

    if title is not None:
        plt.suptitle(title)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def generate_gradcam_for_bscan(
    model: nn.Module,
    roi_volume: np.ndarray,
    bscan_index: int,
    target_layer: nn.Module,
    class_index: int = 1,
    device: str | None = None,
) -> dict[str, Any]:
    """
    Generate Grad-CAM for one B-scan from one ROI volume.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    bscan = roi_volume[int(bscan_index)]
    image_tensor = preprocess_bscan_for_resnet(bscan).to(device)

    gradcam = GradCAM(model, target_layer=target_layer)

    try:
        heatmap = gradcam.generate(image_tensor, class_index=class_index)
    finally:
        gradcam.remove_hooks()

    return {
        "bscan": bscan,
        "heatmap": heatmap,
        "bscan_index": int(bscan_index),
    }


def select_gradcam_examples(
    predictions_df: pd.DataFrame,
    max_examples: int = 10,
    prob_col: str = "barcode_prob",
) -> pd.DataFrame:
    """
    Select high-probability examples for Grad-CAM.

    TODO after clinician labels arrive:
    Replace this with TP/FP/TN/FN selection using label columns.
    """
    if prob_col not in predictions_df.columns:
        raise KeyError(f"Prediction probability column not found: {prob_col}")

    return (
        predictions_df.sort_values(prob_col, ascending=False)
        .head(max_examples)
        .copy()
    )


def export_gradcam_examples(
    processed_dir: str | Path,
    model: nn.Module,
    predictions_df: pd.DataFrame,
    target_layer: nn.Module,
    max_examples: int = 10,
    class_index: int = 1,
    device: str | None = None,
) -> pd.DataFrame:
    """
    Export Grad-CAM figures for selected barcode-positive predictions.

    Expected prediction columns:
    - patient_id
    - file_name
    - bscan_index
    - roi_path OR indirectly recoverable later

    TODO after prediction format is finalized:
    Confirm whether roi_path is saved in slice prediction file. If not,
    merge with preprocessing_qc.csv before calling this.
    """
    dirs = make_gradcam_dirs(processed_dir)

    examples = select_gradcam_examples(
        predictions_df,
        max_examples=max_examples,
    )

    rows = []

    for _, row in examples.iterrows():
        patient_id = str(row["patient_id"])
        file_name = row["file_name"]
        file_stem = Path(file_name).stem
        bscan_index = int(row["bscan_index"])

        if "roi_path" not in row or pd.isna(row["roi_path"]):
            raise KeyError(
                "Grad-CAM export requires `roi_path` in predictions_df. "
                "Merge predictions with preprocessing_qc.csv first."
            )

        roi_volume = load_roi_volume(row["roi_path"])

        out = generate_gradcam_for_bscan(
            model=model,
            roi_volume=roi_volume,
            bscan_index=bscan_index,
            target_layer=target_layer,
            class_index=class_index,
            device=device,
        )

        fig_path = (
            dirs["figures"]
            / patient_id
            / f"{file_stem}_bscan_{bscan_index:03d}_gradcam.png"
        )

        save_gradcam_figure(
            bscan=out["bscan"],
            heatmap=out["heatmap"],
            output_path=fig_path,
            title=f"{patient_id} | {file_name} | B-scan {bscan_index}",
        )

        rows.append(
            {
                "patient_id": patient_id,
                "file_name": file_name,
                "bscan_index": bscan_index,
                "gradcam_path": str(fig_path),
            }
        )

    summary_df = pd.DataFrame(rows)

    summary_path = dirs["figures"] / "gradcam_export_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    return summary_df