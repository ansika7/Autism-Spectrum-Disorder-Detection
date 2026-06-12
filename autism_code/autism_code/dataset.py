"""
dataset.py — Loads pre-split augmented dataset
================================================
Reads existing train / validation / test folder structure directly.
No splitting is performed.

Expected layout:
    data/augmented/
        train/
            autistic/   *.jpg
            control/    *.jpg
        validation/
            autistic/   *.jpg
            control/    *.jpg
        test/
            autistic/   *.jpg
            control/    *.jpg
"""

import logging
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
from PIL import Image

log = logging.getLogger(__name__)

_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


def build_inference_transform(cfg: dict) -> Callable:
    sz   = cfg["dataset"]["image_size"]
    mean = cfg["dataset"].get("norm_mean", [0.485, 0.456, 0.406])
    std  = cfg["dataset"].get("norm_std",  [0.229, 0.224, 0.225])
    return transforms.Compose([
        transforms.Resize((sz, sz)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


class ASDDataset(Dataset):
    def __init__(
        self,
        folder:      str,
        transform:   Optional[Callable] = None,
        class_names: Optional[List[str]] = None,
    ):
        self.folder      = Path(folder)
        self.transform   = transform
        self.class_names = class_names or ["autistic", "control"]
        self.class_to_idx = {c: i for i, c in enumerate(self.class_names)}
        self.samples     = self._discover()

        if not self.samples:
            log.warning(f"No images found in {self.folder}")

        counts = {c: 0 for c in self.class_names}
        for _, lbl in self.samples:
            counts[self.class_names[lbl]] += 1
        log.info(f"[{self.folder.name}] {counts}")

    def _discover(self) -> List[Tuple[Path, int]]:
        out = []
        for cls_name, idx in self.class_to_idx.items():
            cls_dir = self.folder / cls_name
            if not cls_dir.exists():
                log.warning(f"Folder not found: {cls_dir}")
                continue
            for p in sorted(cls_dir.rglob("*")):
                if p.suffix.lower() in _IMG_EXTS:
                    out.append((p, idx))
        return out

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label

    @property
    def targets(self):
        return [lbl for _, lbl in self.samples]

    def class_weights(self) -> torch.Tensor:
        counts = torch.zeros(len(self.class_names))
        for _, lbl in self.samples:
            counts[lbl] += 1
        log.info(f"Class counts: {counts.tolist()}")
        w = 1.0 / counts.clamp(min=1.0)
        return w / w.sum()


def build_dataloaders(cfg: dict) -> Dict[str, DataLoader]:
    ds_cfg    = cfg["dataset"]
    tr_cfg    = cfg["training"]
    root      = Path(ds_cfg["augmented_dir"])
    cls_names = ds_cfg.get("classes", ["autistic", "control"])
    tfm       = build_inference_transform(cfg)

    # Map split names — handles "validation" or "val"
    val_folder = "validation"

    train_ds = ASDDataset(root / "train",      tfm, cls_names)
    val_ds   = ASDDataset(root / val_folder,   tfm, cls_names)
    test_ds  = ASDDataset(root / "test",       tfm, cls_names)

    log.info(
        f"Loaded → train: {len(train_ds)}, "
        f"val: {len(val_ds)}, test: {len(test_ds)}"
    )

    # Weighted sampler for class imbalance
    sampler = None
    if tr_cfg.get("use_weighted_sampler", True) and len(train_ds) > 0:
        cw       = train_ds.class_weights()
        sample_w = torch.tensor([cw[t] for t in train_ds.targets], dtype=torch.float)
        sampler  = WeightedRandomSampler(sample_w, len(sample_w), replacement=True)
        log.info(f"WeightedRandomSampler | class weights: {cw.tolist()}")

    kw = dict(
        num_workers = tr_cfg.get("num_workers", 0),
        pin_memory  = tr_cfg.get("pin_memory", False),
    )

    return {
        "train": DataLoader(train_ds,
                            batch_size=tr_cfg["batch_size"],
                            sampler=sampler,
                            shuffle=(sampler is None),
                            **kw),
        "val":   DataLoader(val_ds,
                            batch_size=tr_cfg["batch_size"],
                            shuffle=False, **kw),
        "test":  DataLoader(test_ds,
                            batch_size=tr_cfg["batch_size"],
                            shuffle=False, **kw),
    }