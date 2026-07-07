from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms


# ---------------------------------------------------------------------
# TODO after clinician labels arrive:
# Confirm final label file path and label column names.
# Current assumed file:
# data/processed/labels/clinician/barcode_labels.csv
# Current assumed label column:
# barcode_volume_status with values: absent / present / unsure
# ---------------------------------------------------------------------

POSITIVE_VALUES = {"present", "positive", "yes", "1", 1}
NEGATIVE_VALUES = {"absent", "negative", "no", "0", 0}


def make_resnet_dirs(processed_dir: str | Path) -> dict[str, Path]:
    processed_dir = Path(processed_dir)

    dirs = {
        "processed": processed_dir,
        "models": processed_dir / "models",
        "predictions": processed_dir / "predictions",
        "qc": processed_dir / "qc",
        "figures": processed_dir / "figures" / "resnet",
    }

    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    return dirs


def load_preprocessing_qc(processed_dir: str | Path) -> pd.DataFrame:
    path = Path(processed_dir) / "qc" / "preprocessing_qc.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing preprocessing QC file: {path}")
    return pd.read_csv(path)


def load_clinician_labels(label_file: str | Path) -> pd.DataFrame:
    label_file = Path(label_file)
    if not label_file.exists():
        raise FileNotFoundError(
            f"Missing clinician label file: {label_file}\n"
            "Place completed labels in data/processed/labels/clinician/."
        )
    return pd.read_csv(label_file)


def normalize_label(value: Any) -> int | None:
    """
    Convert clinician label value to binary.

    Returns
    -------
    1 for barcode-positive
    0 for barcode-negative
    None for unsure/missing/excluded
    """
    if pd.isna(value):
        return None

    value_clean = str(value).strip().lower()

    if value_clean in {str(v).lower() for v in POSITIVE_VALUES}:
        return 1

    if value_clean in {str(v).lower() for v in NEGATIVE_VALUES}:
        return 0

    return None


def build_training_table(
    processed_dir: str | Path,
    label_file: str | Path,
    label_col: str = "barcode_volume_status",
) -> pd.DataFrame:
    """
    Merge preprocessing QC with clinician labels.

    One row corresponds to one E2E volume.

    TODO after clinician labels arrive:
    - Update label_col if clinician CSV uses a different name.
    - Update merge keys if labels include additional identifiers.
    """
    qc_df = load_preprocessing_qc(processed_dir)
    label_df = load_clinician_labels(label_file)

    merged = qc_df.merge(
        label_df,
        on=["patient_id", "file_name"],
        how="inner",
        suffixes=("", "_label"),
    )

    if label_col not in merged.columns:
        raise KeyError(
            f"Label column '{label_col}' not found. "
            f"Available columns: {list(merged.columns)}"
        )

    merged["barcode_label"] = merged[label_col].apply(normalize_label)
    merged = merged[merged["barcode_label"].notna()].copy()
    merged["barcode_label"] = merged["barcode_label"].astype(int)

    merged = merged[merged["preprocess_ok"] == True].copy()

    return merged


def volume_to_slices_table(
    volume_df: pd.DataFrame,
    use_positive_range: bool = True,
) -> pd.DataFrame:
    """
    Expand volume-level rows into B-scan-level training examples.

    If first_positive_bscan / last_positive_bscan are available, positive
    volumes can be expanded so only the positive slice range receives label 1.

    TODO after clinician labels arrive:
    - Confirm whether first_positive_bscan and last_positive_bscan are provided.
    - If labels remain volume-level only, set use_positive_range=False.
    """
    rows = []

    for _, row in volume_df.iterrows():
        n_bscans = int(row["n_bscans"])
        volume_label = int(row["barcode_label"])

        first = row.get("first_positive_bscan", np.nan)
        last = row.get("last_positive_bscan", np.nan)

        has_range = (
            use_positive_range
            and volume_label == 1
            and pd.notna(first)
            and pd.notna(last)
            and str(first).strip() != ""
            and str(last).strip() != ""
        )

        for bidx in range(n_bscans):
            label = volume_label

            if has_range:
                label = int(int(first) <= bidx <= int(last))

            rows.append(
                {
                    "patient_id": row["patient_id"],
                    "file_name": row["file_name"],
                    "roi_path": row["roi_path"],
                    "bscan_index": bidx,
                    "label": label,
                }
            )

    return pd.DataFrame(rows)


def patient_split(
    df: pd.DataFrame,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Patient-level split to avoid leakage.
    """
    rng = np.random.default_rng(seed)

    patients = np.array(sorted(df["patient_id"].unique()))
    rng.shuffle(patients)

    n = len(patients)
    n_train = int(train_frac * n)
    n_val = int(val_frac * n)

    train_patients = set(patients[:n_train])
    val_patients = set(patients[n_train : n_train + n_val])
    test_patients = set(patients[n_train + n_val :])

    train_df = df[df["patient_id"].isin(train_patients)].copy()
    val_df = df[df["patient_id"].isin(val_patients)].copy()
    test_df = df[df["patient_id"].isin(test_patients)].copy()

    return train_df, val_df, test_df


class BarcodeSliceDataset(Dataset):
    """
    Dataset for B-scan-level ResNet training.

    Each item loads one ROI volume from .npy and returns one B-scan.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        image_size: int = 224,
        training: bool = False,
    ):
        self.df = df.reset_index(drop=True)
        self.image_size = image_size
        self.training = training

        base_transforms = [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]

        aug_transforms = [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]

        self.transform = transforms.Compose(
            aug_transforms if training else base_transforms
        )

    def __len__(self) -> int:
        return len(self.df)

    def _load_bscan(self, roi_path: str | Path, bscan_index: int) -> np.ndarray:
        volume = np.load(roi_path)
        bscan = volume[int(bscan_index)]

        bscan = bscan.astype(np.float32)
        bscan = np.nan_to_num(bscan)

        # Rescale each B-scan to uint8 for PIL/ResNet input.
        bmin = bscan.min()
        bmax = bscan.max()
        bscan = (bscan - bmin) / (bmax - bmin + 1e-8)
        bscan = (255 * bscan).astype(np.uint8)

        return bscan

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.df.iloc[idx]

        bscan = self._load_bscan(row["roi_path"], row["bscan_index"])

        img = Image.fromarray(bscan).convert("RGB")
        x = self.transform(img)

        y = torch.tensor(int(row["label"]), dtype=torch.long)

        return {
            "image": x,
            "label": y,
            "patient_id": row["patient_id"],
            "file_name": row["file_name"],
            "bscan_index": int(row["bscan_index"]),
        }


def build_resnet50_binary(
    c8_checkpoint_path: str | Path | None = None,
    freeze_backbone: bool = True,
) -> nn.Module:
    """
    Build ResNet50 binary classifier.

    If c8_checkpoint_path is provided, load OCT-C8 weights first, then replace
    the final classification head with a binary head.

    TODO after checkpoint finalization:
    - Confirm checkpoint format: raw state_dict vs dict with 'model_state_dict'.
    - Confirm whether C8 model used fc output size 8.
    """
    model = models.resnet50(weights=None)

    # Temporary C8 head before loading C8 checkpoint.
    model.fc = nn.Linear(model.fc.in_features, 8)

    if c8_checkpoint_path is not None:
        checkpoint = torch.load(c8_checkpoint_path, map_location="cpu")

        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint

        model.load_state_dict(state_dict, strict=False)

    # Replace C8 disease head with binary barcode head.
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, 2)

    if freeze_backbone:
        for name, param in model.named_parameters():
            param.requires_grad = False

        for param in model.fc.parameters():
            param.requires_grad = True

    return model


def unfreeze_layer4(model: nn.Module) -> nn.Module:
    """
    Unfreeze final ResNet block plus classifier.
    """
    for param in model.layer4.parameters():
        param.requires_grad = True

    for param in model.fc.parameters():
        param.requires_grad = True

    return model


def make_dataloaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    batch_size: int = 32,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_ds = BarcodeSliceDataset(train_df, training=True)
    val_ds = BarcodeSliceDataset(val_df, training=False)
    test_ds = BarcodeSliceDataset(test_df, training=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, val_loader, test_loader


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: str,
) -> dict[str, float]:
    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for batch in loader:
        x = batch["image"].to(device)
        y = batch["label"].to(device)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        pred = logits.argmax(dim=1)

        correct += (pred == y).sum().item()
        total += x.size(0)

    return {
        "loss": total_loss / max(total, 1),
        "accuracy": correct / max(total, 1),
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: str,
) -> dict[str, float]:
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    for batch in loader:
        x = batch["image"].to(device)
        y = batch["label"].to(device)

        logits = model(x)
        loss = criterion(logits, y)

        total_loss += loss.item() * x.size(0)
        pred = logits.argmax(dim=1)

        correct += (pred == y).sum().item()
        total += x.size(0)

    return {
        "loss": total_loss / max(total, 1),
        "accuracy": correct / max(total, 1),
    }


@torch.no_grad()
def predict_slices(
    model: nn.Module,
    loader: DataLoader,
    device: str,
) -> pd.DataFrame:
    model.eval()

    rows = []

    for batch in loader:
        x = batch["image"].to(device)

        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        pred = probs.argmax(dim=1)

        for i in range(x.size(0)):
            rows.append(
                {
                    "patient_id": batch["patient_id"][i],
                    "file_name": batch["file_name"][i],
                    "bscan_index": int(batch["bscan_index"][i]),
                    "barcode_prob": float(probs[i, 1].cpu()),
                    "barcode_pred": int(pred[i].cpu()),
                }
            )

    return pd.DataFrame(rows)


def aggregate_volume_predictions(slice_pred_df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse slice predictions to volume-level predictions.
    """
    volume_df = (
        slice_pred_df.groupby(["patient_id", "file_name"])
        .agg(
            mean_barcode_prob=("barcode_prob", "mean"),
            max_barcode_prob=("barcode_prob", "max"),
            positive_bscan_count=("barcode_pred", "sum"),
            n_bscans=("barcode_pred", "size"),
        )
        .reset_index()
    )

    volume_df["positive_bscan_fraction"] = (
        volume_df["positive_bscan_count"] / volume_df["n_bscans"]
    )

    volume_df["barcode_volume_pred"] = (
        volume_df["positive_bscan_fraction"] > 0
    ).astype(int)

    # TODO after clinician labels arrive:
    # Tune this aggregation rule. Possible definitions:
    # - any positive B-scan
    # - at least k positive B-scans
    # - positive_bscan_fraction >= threshold
    # - max probability >= threshold

    return volume_df


def run_resnet_finetuning(
    processed_dir: str | Path,
    label_file: str | Path,
    c8_checkpoint_path: str | Path | None = None,
    label_col: str = "barcode_volume_status",
    use_positive_range: bool = True,
    batch_size: int = 32,
    epochs: int = 5,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    unfreeze_final_block: bool = True,
    device: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    End-to-end fine-tuning wrapper.

    TODO after clinician labels arrive:
    - Verify label column and positive/negative values.
    - Decide whether to use positive B-scan ranges.
    - Tune epochs/lr/unfreezing after initial run.
    """
    processed_dir = Path(processed_dir)
    dirs = make_resnet_dirs(processed_dir)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    volume_df = build_training_table(
        processed_dir=processed_dir,
        label_file=label_file,
        label_col=label_col,
    )

    slice_df = volume_to_slices_table(
        volume_df,
        use_positive_range=use_positive_range,
    )

    train_df, val_df, test_df = patient_split(slice_df)

    train_loader, val_loader, test_loader = make_dataloaders(
        train_df,
        val_df,
        test_df,
        batch_size=batch_size,
    )

    model = build_resnet50_binary(
        c8_checkpoint_path=c8_checkpoint_path,
        freeze_backbone=True,
    )

    if unfreeze_final_block:
        model = unfreeze_layer4(model)

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=weight_decay,
    )

    history = []

    for epoch in range(1, epochs + 1):
        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
        )

        val_metrics = evaluate(
            model,
            val_loader,
            criterion,
            device,
        )

        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
        }

        history.append(row)

        print(
            f"Epoch {epoch}/{epochs} | "
            f"train_loss={row['train_loss']:.4f} | "
            f"train_acc={row['train_accuracy']:.4f} | "
            f"val_loss={row['val_loss']:.4f} | "
            f"val_acc={row['val_accuracy']:.4f}"
        )

    history_df = pd.DataFrame(history)

    model_path = dirs["models"] / "barcode_resnet.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "history": history,
        },
        model_path,
    )

    slice_pred_df = predict_slices(model, test_loader, device)
    volume_pred_df = aggregate_volume_predictions(slice_pred_df)

    slice_pred_path = dirs["predictions"] / "barcode_slice_predictions.csv"
    volume_pred_path = dirs["predictions"] / "barcode_volume_predictions.csv"
    history_path = dirs["qc"] / "resnet_training_summary.csv"

    slice_pred_df.to_csv(slice_pred_path, index=False)
    volume_pred_df.to_csv(volume_pred_path, index=False)
    history_df.to_csv(history_path, index=False)

    print("\nSaved:")
    print(f"- Model: {model_path}")
    print(f"- Slice predictions: {slice_pred_path}")
    print(f"- Volume predictions: {volume_pred_path}")
    print(f"- Training history: {history_path}")

    return history_df, volume_pred_df