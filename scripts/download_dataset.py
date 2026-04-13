import argparse
import os
import shutil
import zipfile
from pathlib import Path

import kagglehub


TARGET_ROOT = Path("dataset")
TARGET_DYS = TARGET_ROOT / "dyslexic"
TARGET_NON = TARGET_ROOT / "non_dyslexic"
DEFAULT_PASSWORD = "WanAsy321"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _find_class_folders(dataset_path: Path):
    folders = []
    for split in ("train", "test"):
        for klass in ("corrected", "normal", "reversal"):
            pattern = f"**/gambo/{split}/{klass}"
            for p in dataset_path.glob(pattern):
                if p.is_dir():
                    folders.append(p)
    return folders


def _class_from_path(path: Path, corrected_as: str):
    low = str(path).lower()

    if "\\normal\\" in low or "/normal/" in low:
        return "non_dyslexic"
    if "\\reversal\\" in low or "/reversal/" in low:
        return "dyslexic"
    if "\\corrected\\" in low or "/corrected/" in low:
        return corrected_as

    # Fallback for other naming variants.
    if "dyslexic" in low:
        return "dyslexic"
    if "non" in low and "dys" in low:
        return "non_dyslexic"
    if "normal" in low:
        return "non_dyslexic"
    return None


def _extract_archives(dataset_path: Path, password: str):
    archives = list(dataset_path.rglob("*.zip"))
    if not archives:
        return dataset_path

    extract_root = dataset_path / "_extracted"
    extract_root.mkdir(parents=True, exist_ok=True)
    pwd = password.encode("utf-8") if password else None

    extracted_count = 0
    for archive in archives:
        out_dir = extract_root / archive.stem
        out_dir.mkdir(parents=True, exist_ok=True)
        try:
            with zipfile.ZipFile(archive) as zf:
                if pwd:
                    zf.extractall(path=out_dir, pwd=pwd)
                else:
                    zf.extractall(path=out_dir)
            extracted_count += 1
        except RuntimeError as exc:
            # RuntimeError is raised for bad password in encrypted zip entries.
            print(f"Skipping encrypted archive (bad password): {archive} -> {exc}")
        except zipfile.BadZipFile as exc:
            print(f"Skipping invalid zip archive: {archive} -> {exc}")

    if extracted_count:
        print(f"Extracted {extracted_count} archive(s) into: {extract_root}")
    return dataset_path


def _reset_target_dirs():
    if TARGET_DYS.exists():
        shutil.rmtree(TARGET_DYS)
    if TARGET_NON.exists():
        shutil.rmtree(TARGET_NON)
    TARGET_DYS.mkdir(parents=True, exist_ok=True)
    TARGET_NON.mkdir(parents=True, exist_ok=True)


def main(password: str, corrected_as: str, reset_target: bool):
    if reset_target:
        _reset_target_dirs()

    TARGET_DYS.mkdir(parents=True, exist_ok=True)
    TARGET_NON.mkdir(parents=True, exist_ok=True)

    # Download latest version from KaggleHub cache.
    dataset_path = Path(
        kagglehub.dataset_download("drizasazanitaisa/dyslexia-handwriting-dataset")
    )
    print(f"Downloaded dataset to: {dataset_path}")

    _extract_archives(dataset_path, password)

    source_folders = _find_class_folders(dataset_path)
    if source_folders:
        print("Detected Gambo folder layout:")
        for folder in source_folders:
            print(f"- {folder}")

    copied = 0
    dys_count = 0
    non_count = 0

    file_iter = (
        (p for src in source_folders for p in src.rglob("*"))
        if source_folders
        else dataset_path.rglob("*")
    )

    for file_path in file_iter:
        if not file_path.is_file() or file_path.suffix.lower() not in IMAGE_EXTS:
            continue

        klass = _class_from_path(file_path, corrected_as)
        if klass is None:
            continue

        target_dir = TARGET_DYS if klass == "dyslexic" else TARGET_NON
        target_file = target_dir / file_path.name

        if target_file.exists():
            stem = target_file.stem
            suffix = target_file.suffix
            idx = 1
            while True:
                candidate = target_dir / f"{stem}_{idx}{suffix}"
                if not candidate.exists():
                    target_file = candidate
                    break
                idx += 1

        shutil.copy2(file_path, target_file)
        copied += 1
        if klass == "dyslexic":
            dys_count += 1
        else:
            non_count += 1

    print(f"Copied {copied} images into {TARGET_ROOT.resolve()}")
    print(f"dyslexic: {dys_count}")
    print(f"non_dyslexic: {non_count}")
    print("Expected structure:")
    print("dataset/dyslexic")
    print("dataset/non_dyslexic")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--password",
        type=str,
        default=os.getenv("DATASET_PASSWORD", DEFAULT_PASSWORD),
        help="Password for encrypted dataset zip archives.",
    )
    parser.add_argument(
        "--corrected-as",
        type=str,
        choices=["dyslexic", "non_dyslexic"],
        default="dyslexic",
        help="How to map the 'Corrected' folder into binary labels.",
    )
    parser.add_argument(
        "--reset-target",
        action="store_true",
        help="Delete dataset/dyslexic and dataset/non_dyslexic before copying.",
    )
    args = parser.parse_args()

    main(args.password, args.corrected_as, args.reset_target)
