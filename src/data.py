import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def parse_split_line(line: str) -> Tuple[str, int]:
    """
    Accepts:
      - "path label"
      - "label path"
    Returns (path, label).
    """
    parts = line.strip().split()
    if len(parts) < 2:
        raise ValueError(f"Bad split line: {line!r}")

    # Try "path label"
    try:
        label = int(parts[-1])
        path = " ".join(parts[:-1])
        return path, label
    except ValueError:
        pass

    # Try "label path"
    try:
        label = int(parts[0])
        path = " ".join(parts[1:])
        return path, label
    except ValueError:
        raise ValueError(f"Cannot parse split line: {line!r}")

def resolve_to_text_path(qs_root: Path, rel_path: str) -> Path:
    """
    If rel_path points to image (tif/png/jpg), map to the QS-OCR-Large text file if possible.
    If it already ends with .txt, use as is.
    """
    #
    # p = Path(rel_path)
    # if p.suffix.lower() != ".txt":
    #     p = p.with_suffix(".txt")
    # return qs_root / p

    # Normalize slashes
    rel_path = rel_path.replace("\\", "/").lstrip("./")

    p = Path(rel_path)
    # Map image -> txt
    if p.suffix.lower() != ".txt":
        p = p.with_suffix(".txt")

    return qs_root / p

class RVLCDIPOCRTextDataset(Dataset):
    """
    In-distribution dataset: (text, label).
    Uses QS-OCR-Large directory tree.
    """
    def __init__(
        self,
        qs_root: Path,
        split_file: Path,
        tokenizer_name: str,
        max_length: int,
        debug_samples: Optional[int] = None,
        allowed_labels: Optional[set] = None,
    ):
        self.qs_root = qs_root
        self.items: List[Tuple[Path, int]] = []

        with split_file.open("r", encoding="utf-8", errors="ignore") as f:
            lines = [ln for ln in f if ln.strip()]

        for ln in lines:
            rel, label = parse_split_line(ln)
            if allowed_labels is not None and label not in allowed_labels:
                continue
            txt_path = resolve_to_text_path(qs_root, rel)
            self.items.append((txt_path, label))

        if debug_samples is not None:
            self.items = self.items[:debug_samples]

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        txt_path, label = self.items[idx]
        text = txt_path.read_text(encoding="utf-8", errors="ignore")

        enc = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        enc = {k: v.squeeze(0) for k, v in enc.items()}
        enc["labels"] = torch.tensor(label, dtype=torch.long)
        enc["path_idx"] = torch.tensor(idx, dtype=torch.long)
        return enc

class RVLCDIPOODTextDataset(Dataset):
    """
    Out-of-distribution dataset: (text, label=-1).
    Loads *.txt files from rvl-cdip-o-text.
    """
    def __init__(
        self,
        ood_text_dir: Path,
        tokenizer_name: str,
        max_length: int,
        debug_samples: Optional[int] = None,
    ):
        txt_files = sorted(ood_text_dir.glob("*.txt"))
        if debug_samples is not None:
            txt_files = txt_files[:debug_samples]
        self.items = txt_files

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        txt_path = self.items[idx]
        text = txt_path.read_text(encoding="utf-8", errors="ignore")

        enc = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        enc = {k: v.squeeze(0) for k, v in enc.items()}
        enc["labels"] = torch.tensor(-1, dtype=torch.long)
        enc["path_idx"] = torch.tensor(idx, dtype=torch.long)
        return enc