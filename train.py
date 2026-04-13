import argparse
import random
from pathlib import Path

import kagglehub
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm
from torchvision import datasets, transforms

from model_utils import IMAGENET_MEAN, IMAGENET_STD, build_model


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_transforms(augment: bool):
    transform_list = [transforms.Resize((224, 224))]
    if augment:
        transform_list.extend(
            [
                transforms.RandomRotation(degrees=10),
                transforms.ColorJitter(brightness=0.2),
            ]
        )

    transform_list.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return transforms.Compose(transform_list)


class BinaryPathDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, label = self.samples[idx]
        with Image.open(image_path) as img:
            image = img.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label


def _find_split_dir(base_dir: Path, split_name: str):
    split_lower = split_name.lower()
    for p in base_dir.rglob("*"):
        if p.is_dir() and p.name.lower() == split_lower:
            return p
    return None


def _find_original_split_root(data_dir: Path):
    train_dir = _find_split_dir(data_dir, "train")
    test_dir = _find_split_dir(data_dir, "test")

    if train_dir and test_dir:
        return train_dir, test_dir
    return None, None


def _collect_binary_samples(split_dir: Path, corrected_as: str, class_to_idx: dict):
    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    samples = []

    for image_path in split_dir.rglob("*"):
        if not image_path.is_file() or image_path.suffix.lower() not in image_exts:
            continue

        low = str(image_path.parent).lower()
        if "normal" in low:
            binary_class = "non_dyslexic"
        elif "reversal" in low:
            binary_class = "dyslexic"
        elif "corrected" in low:
            binary_class = corrected_as
        elif "dyslexic" in low:
            binary_class = "dyslexic"
        elif "non_dyslexic" in low:
            binary_class = "non_dyslexic"
        else:
            continue

        samples.append((str(image_path), class_to_idx[binary_class]))

    return samples


def _build_datasets(args):
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory '{data_dir}' not found.")

    class_to_idx = {"dyslexic": 0, "non_dyslexic": 1}

    if args.split_mode == "original":
        train_dir, test_dir = _find_original_split_root(data_dir)

        if train_dir is None or test_dir is None:
            kaggle_root = Path(
                kagglehub.dataset_download("drizasazanitaisa/dyslexia-handwriting-dataset")
            )
            train_dir, test_dir = _find_original_split_root(kaggle_root)

        if train_dir is None or test_dir is None:
            raise ValueError(
                "Original split not found. Provide a folder containing Train/Test directories or use --split-mode random."
            )

        train_samples = _collect_binary_samples(train_dir, args.corrected_as, class_to_idx)
        val_samples = _collect_binary_samples(test_dir, args.corrected_as, class_to_idx)

        if not train_samples or not val_samples:
            raise ValueError(
                "No images found under original split after class mapping."
            )

        train_dataset = BinaryPathDataset(
            train_samples, transform=get_transforms(args.augment)
        )
        val_dataset = BinaryPathDataset(val_samples, transform=get_transforms(False))
        classes = ["dyslexic", "non_dyslexic"]
        return train_dataset, val_dataset, class_to_idx, classes

    base_dataset = datasets.ImageFolder(root=str(data_dir))
    if len(base_dataset.classes) != 2:
        raise ValueError(f"Expected exactly 2 classes, got {base_dataset.classes}.")

    train_size = int(0.8 * len(base_dataset))
    val_size = len(base_dataset) - train_size

    train_subset, val_subset = random_split(
        base_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed),
    )

    train_subset.dataset = datasets.ImageFolder(
        root=str(data_dir), transform=get_transforms(args.augment)
    )
    val_subset.dataset = datasets.ImageFolder(
        root=str(data_dir), transform=get_transforms(False)
    )

    return train_subset, val_subset, base_dataset.class_to_idx, base_dataset.classes


def train_one_epoch(model, loader, criterion, optimizer, device, class_to_idx, epoch):
    model.train()
    total_loss = 0.0
    y_true = []
    y_pred = []

    bar = tqdm(loader, desc=f"Epoch {epoch} Train", leave=False)

    for images, labels in bar:
        images = images.to(device, non_blocking=True)
        labels = labels.float().unsqueeze(1).to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        probs = torch.sigmoid(logits).squeeze(1)
        preds = (probs >= 0.5).long()

        y_true.extend(labels.squeeze(1).long().detach().cpu().tolist())
        y_pred.extend(preds.detach().cpu().tolist())

        total_loss += loss.item() * images.size(0)
        avg_loss = total_loss / max(1, len(y_true))
        dys_f1 = f1_score(
            y_true,
            y_pred,
            pos_label=class_to_idx["dyslexic"],
            zero_division=0,
        )
        acc = accuracy_score(y_true, y_pred)
        bar.set_postfix(loss=f"{avg_loss:.4f}", acc=f"{acc:.4f}", f1=f"{dys_f1:.4f}")

    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(y_true, y_pred)
    dys_f1 = f1_score(
        y_true,
        y_pred,
        pos_label=class_to_idx["dyslexic"],
        zero_division=0,
    )
    return avg_loss, acc, dys_f1


@torch.no_grad()
def evaluate(model, loader, criterion, device, class_to_idx, epoch):
    model.eval()
    total_loss = 0.0
    y_true = []
    y_pred = []

    bar = tqdm(loader, desc=f"Epoch {epoch} Val", leave=False)

    for images, labels in bar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, labels.float().unsqueeze(1))
        total_loss += loss.item() * images.size(0)

        probs = torch.sigmoid(logits).squeeze(1)
        preds = (probs >= 0.5).long()

        y_true.extend(labels.cpu().tolist())
        y_pred.extend(preds.cpu().tolist())

        avg_loss = total_loss / max(1, len(y_true))
        dys_f1 = f1_score(
            y_true,
            y_pred,
            pos_label=class_to_idx["dyslexic"],
            zero_division=0,
        )
        acc = accuracy_score(y_true, y_pred)
        bar.set_postfix(loss=f"{avg_loss:.4f}", acc=f"{acc:.4f}", f1=f"{dys_f1:.4f}")

    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(
        y_true,
        y_pred,
        pos_label=class_to_idx["dyslexic"],
        zero_division=0,
    )
    return avg_loss, acc, f1


def main(args):
    seed_everything(args.seed)

    train_subset, val_subset, class_to_idx, classes = _build_datasets(args)

    train_loader = DataLoader(
        train_subset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(pretrained=True).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_f1 = -1.0
    artifact_dir = Path(args.output_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = artifact_dir / "convnext_mvp.pt"

    print(f"Classes: {classes}")
    print(f"Class mapping: {class_to_idx}")
    print(f"Split mode: {args.split_mode}")
    print(f"Corrected mapping: {args.corrected_as}")
    print(f"Train size: {len(train_subset)}, Val size: {len(val_subset)}")
    print(f"Device: {device}")

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc, train_f1 = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            class_to_idx,
            epoch,
        )
        val_loss, val_acc, val_f1 = evaluate(
            model,
            val_loader,
            criterion,
            device,
            class_to_idx,
            epoch,
        )

        print(
            f"Epoch {epoch:02d}/{args.epochs} "
            f"train_loss={train_loss:.4f} "
            f"train_acc={train_acc:.4f} "
            f"train_f1={train_f1:.4f} "
            f"val_loss={val_loss:.4f} "
            f"val_acc={val_acc:.4f} "
            f"val_f1={val_f1:.4f}"
        )

        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "class_to_idx": class_to_idx,
                    "best_val_f1": float(best_f1),
                    "args": vars(args),
                },
                ckpt_path,
            )
            print(f"Saved best model -> {ckpt_path}")

    print("Training finished.")
    print(f"Best validation F1: {best_f1:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="dataset")
    parser.add_argument("--output-dir", type=str, default="artifacts")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--augment", action="store_true")
    parser.add_argument(
        "--split-mode",
        type=str,
        choices=["original", "random"],
        default="original",
    )
    parser.add_argument(
        "--corrected-as",
        type=str,
        choices=["dyslexic", "non_dyslexic"],
        default="dyslexic",
    )

    main(parser.parse_args())
