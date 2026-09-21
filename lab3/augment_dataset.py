import argparse
import json
import random
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Dict, List, Sequence, Tuple

from PIL import Image, ImageEnhance, ImageOps


VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass
class AugmentationConfig:
    rotation_range: Tuple[float, float] = (-30.0, 30.0)
    scale_range: Tuple[float, float] = (0.9, 1.1)
    crop_scale_range: Tuple[float, float] = (0.8, 1.0)
    color_factor_range: Tuple[float, float] = (0.8, 1.2)
    horizontal_flip_probability: float = 1.0


class MangoLeafAugmentor:
    def __init__(self, seed: int, config: AugmentationConfig):
        self.seed = seed
        self.config = config

    def _open_rgb(self, image_path: Path) -> Image.Image:
        with Image.open(image_path) as img:
            return img.convert("RGB")

    def _center_crop_or_pad_to_size(self, image: Image.Image, target_size: Tuple[int, int]) -> Image.Image:
        target_w, target_h = target_size
        w, h = image.size

        # Crop when larger than target size.
        left = max((w - target_w) // 2, 0)
        top = max((h - target_h) // 2, 0)
        right = min(left + target_w, w)
        bottom = min(top + target_h, h)
        cropped = image.crop((left, top, right, bottom))

        # Pad when smaller than target size to avoid geometric distortion.
        if cropped.size != target_size:
            pad_w = target_w - cropped.size[0]
            pad_h = target_h - cropped.size[1]
            padding = (
                max(pad_w // 2, 0),
                max(pad_h // 2, 0),
                max(pad_w - (pad_w // 2), 0),
                max(pad_h - (pad_h // 2), 0),
            )
            cropped = ImageOps.expand(cropped, border=padding, fill=(0, 0, 0))

        return cropped

    def rotate(self, image: Image.Image, rng: random.Random) -> Image.Image:
        angle = rng.uniform(*self.config.rotation_range)
        return image.rotate(angle, resample=Image.Resampling.BICUBIC, expand=False, fillcolor=(0, 0, 0))

    def hflip(self, image: Image.Image, rng: random.Random) -> Image.Image:
        if rng.random() <= self.config.horizontal_flip_probability:
            return ImageOps.mirror(image)
        return image.copy()

    def scale(self, image: Image.Image, rng: random.Random) -> Image.Image:
        original_size = image.size
        factor = rng.uniform(*self.config.scale_range)
        new_w = max(2, int(original_size[0] * factor))
        new_h = max(2, int(original_size[1] * factor))
        scaled = image.resize((new_w, new_h), resample=Image.Resampling.BICUBIC)
        return self._center_crop_or_pad_to_size(scaled, original_size)

    def random_crop(self, image: Image.Image, rng: random.Random) -> Image.Image:
        w, h = image.size
        scale = rng.uniform(*self.config.crop_scale_range)
        crop_w = max(2, int(w * scale))
        crop_h = max(2, int(h * scale))

        max_left = max(w - crop_w, 0)
        max_top = max(h - crop_h, 0)
        left = rng.randint(0, max_left) if max_left > 0 else 0
        top = rng.randint(0, max_top) if max_top > 0 else 0

        cropped = image.crop((left, top, left + crop_w, top + crop_h))
        return cropped.resize((w, h), resample=Image.Resampling.BICUBIC)

    def color_jitter(self, image: Image.Image, rng: random.Random) -> Image.Image:
        lo, hi = self.config.color_factor_range
        out = image.copy()
        out = ImageEnhance.Brightness(out).enhance(rng.uniform(lo, hi))
        out = ImageEnhance.Contrast(out).enhance(rng.uniform(lo, hi))
        out = ImageEnhance.Color(out).enhance(rng.uniform(lo, hi))
        return out

    def get_augmentation_ops(self) -> Dict[str, Callable[[Image.Image, random.Random], Image.Image]]:
        return {
            "rotation": self.rotate,
            "horizontal_flip": self.hflip,
            "scaling": self.scale,
            "random_crop": self.random_crop,
            "color_adjustment": self.color_jitter,
        }

    def apply(self, image_path: Path, aug_name: str, variant_index: int) -> Image.Image:
        image = self._open_rgb(image_path)
        # Deterministic per-file/augmentation/variant randomness for reproducibility.
        local_seed = hash((str(image_path), aug_name, variant_index, self.seed)) & 0xFFFFFFFF
        rng = random.Random(local_seed)
        op = self.get_augmentation_ops()[aug_name]
        return op(image, rng)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def list_class_dirs(dataset_root: Path) -> List[Path]:
    return sorted([p for p in dataset_root.iterdir() if p.is_dir() and not p.name.startswith(".")])


def list_images(class_dir: Path) -> List[Path]:
    return sorted([p for p in class_dir.rglob("*") if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS])


def split_images(images: Sequence[Path], train_ratio: float, val_ratio: float, test_ratio: float, seed: int) -> Dict[str, List[Path]]:
    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-9:
        raise ValueError("train/val/test ratios must sum to 1.0")

    images = list(images)
    rng = random.Random(seed)
    rng.shuffle(images)

    n = len(images)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    n_test = n - n_train - n_val

    return {
        "train": images[:n_train],
        "val": images[n_train : n_train + n_val],
        "test": images[n_train + n_val : n_train + n_val + n_test],
    }


def copy_originals(split_map: Dict[str, Dict[str, List[Path]]], output_root: Path) -> None:
    for split, class_to_images in split_map.items():
        for class_name, images in class_to_images.items():
            class_out = output_root / split / "original" / class_name
            ensure_dir(class_out)
            for image_path in images:
                dst = class_out / image_path.name
                shutil.copy2(image_path, dst)


def augment_split(
    split: str,
    class_to_images: Dict[str, List[Path]],
    output_root: Path,
    augmentor: MangoLeafAugmentor,
    variants_per_augmentation: int,
    augment_splits: Sequence[str],
) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    if split not in augment_splits:
        return counts

    ops = augmentor.get_augmentation_ops()
    for aug_name in ops:
        counts[aug_name] = 0

    for class_name, images in class_to_images.items():
        for image_path in images:
            stem = image_path.stem
            for aug_name in ops:
                class_out = output_root / split / aug_name / class_name
                ensure_dir(class_out)
                for variant_idx in range(variants_per_augmentation):
                    out_img = augmentor.apply(image_path, aug_name, variant_idx)
                    out_name = f"{stem}__{aug_name}__v{variant_idx + 1:02d}.jpg"
                    out_img.save(class_out / out_name, format="JPEG", quality=95)
                    counts[aug_name] += 1

    return counts


def build_split_map(
    dataset_root: Path,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Dict[str, Dict[str, List[Path]]]:
    split_map: Dict[str, Dict[str, List[Path]]] = {"train": {}, "val": {}, "test": {}}

    for class_dir in list_class_dirs(dataset_root):
        images = list_images(class_dir)
        if not images:
            continue

        per_class_split = split_images(images, train_ratio, val_ratio, test_ratio, seed=seed)
        for split_name in split_map:
            split_map[split_name][class_dir.name] = per_class_split[split_name]

    return split_map


def summarize_counts(split_map: Dict[str, Dict[str, List[Path]]]) -> Dict[str, Dict[str, int]]:
    out: Dict[str, Dict[str, int]] = {}
    for split, class_to_images in split_map.items():
        out[split] = {class_name: len(paths) for class_name, paths in class_to_images.items()}
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create split-aware image augmentation folders for MangoLeafBD dataset")
    parser.add_argument("--input", required=True, help="Path to original dataset root")
    parser.add_argument("--output", required=True, help="Path to output root folder")
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--variants-per-augmentation", type=int, default=1)
    parser.add_argument(
        "--augment-splits",
        nargs="+",
        default=["train"],
        choices=["train", "val", "test"],
        help="Choose which splits receive augmented images (default: train only)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    dataset_root = Path(args.input).resolve()
    output_root = Path(args.output).resolve()

    if not dataset_root.exists() or not dataset_root.is_dir():
        raise FileNotFoundError(f"Input dataset root not found: {dataset_root}")

    ensure_dir(output_root)

    config = AugmentationConfig()
    augmentor = MangoLeafAugmentor(seed=args.seed, config=config)

    split_map = build_split_map(
        dataset_root=dataset_root,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    copy_originals(split_map, output_root)

    augmentation_counts: Dict[str, Dict[str, int]] = {}
    for split_name, class_to_images in split_map.items():
        augmentation_counts[split_name] = augment_split(
            split=split_name,
            class_to_images=class_to_images,
            output_root=output_root,
            augmentor=augmentor,
            variants_per_augmentation=args.variants_per_augmentation,
            augment_splits=args.augment_splits,
        )

    summary = {
        "input_root": str(dataset_root),
        "output_root": str(output_root),
        "split_ratios": {"train": args.train_ratio, "val": args.val_ratio, "test": args.test_ratio},
        "seed": args.seed,
        "variants_per_augmentation": args.variants_per_augmentation,
        "augment_splits": args.augment_splits,
        "augmentation_config": asdict(config),
        "original_split_counts": summarize_counts(split_map),
        "augmentation_image_counts": augmentation_counts,
        "augmentation_types": list(augmentor.get_augmentation_ops().keys()),
    }

    summary_path = output_root / "augmentation_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Augmentation completed.")
    print(f"Output folder: {output_root}")
    print(f"Summary file:  {summary_path}")


if __name__ == "__main__":
    main()
