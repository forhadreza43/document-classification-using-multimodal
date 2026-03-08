from pathlib import Path
from typing import List, Tuple, Optional, Dict

import torch
from torch.utils.data import Dataset
from PIL import Image
from transformers import AutoProcessor


def parse_split_line(line: str) -> Tuple[str, int]:
    parts = line.strip().split()
    if len(parts) < 2:
        raise ValueError(f"Bad split line: {line!r}")
    rel_path = parts[0].replace("\\", "/").lstrip("./")
    label = int(parts[1])
    return rel_path, label


def normalize_box(box, width: int, height: int):
    x1, y1, x2, y2 = box
    return [
        int(1000 * x1 / max(width, 1)),
        int(1000 * y1 / max(height, 1)),
        int(1000 * x2 / max(width, 1)),
        int(1000 * y2 / max(height, 1)),
    ]


def load_words_and_boxes(tsv_path: Path) -> Tuple[List[str], List[List[int]]]:
    words = []
    boxes = []
    with tsv_path.open("r", encoding="utf-8", errors="ignore") as f:
        lines = [ln.strip() for ln in f if ln.strip()]

    # Optional header support
    start_idx = 1 if lines and lines[0].lower().startswith("word") else 0

    for ln in lines[start_idx:]:
        parts = ln.split("\t")
        if len(parts) != 5:
            continue
        word, x1, y1, x2, y2 = parts
        words.append(word)
        boxes.append([int(x1), int(y1), int(x2), int(y2)])

    return words, boxes


class RVLCDIPLayoutLMv3Dataset(Dataset):
    """
    Multimodal dataset for LayoutLMv3:
      - image from rvl-cdip
      - words from OCR TSV
      - word boxes from OCR TSV
      - label from split file
    """
    def __init__(
        self,
        rvl_root: Path,
        ocr_box_root: Path,
        split_file: Path,
        processor_name: str = "microsoft/layoutlmv3-base",
        max_length: int = 512,
        debug_samples: Optional[int] = None,
        allowed_labels: Optional[set] = None,
    ):
        self.rvl_root = rvl_root
        self.ocr_box_root = ocr_box_root
        self.items = []

        with split_file.open("r", encoding="utf-8", errors="ignore") as f:
            lines = [ln for ln in f if ln.strip()]

        for ln in lines:
            rel_path, label = parse_split_line(ln)
            if allowed_labels is not None and label not in allowed_labels:
                continue

            img_path = rvl_root / rel_path
            tsv_path = ocr_box_root / Path(rel_path).with_suffix(".tsv")

            if img_path.exists() and tsv_path.exists():
                self.items.append((img_path, tsv_path, label))

        if debug_samples is not None:
            self.items = self.items[:debug_samples]

        self.processor = AutoProcessor.from_pretrained(processor_name, apply_ocr=False)
        self.max_length = max_length

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        img_path, tsv_path, label = self.items[idx]

        image = Image.open(img_path).convert("RGB")
        width, height = image.size

        words, boxes_px = load_words_and_boxes(tsv_path)
        boxes = [normalize_box(b, width, height) for b in boxes_px]

        # Fallback if OCR file is empty
        if len(words) == 0:
            words = ["[EMPTY]"]
            boxes = [[0, 0, 1, 1]]

        enc = self.processor(
            image,
            words,
            boxes=boxes,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        enc = {k: v.squeeze(0) for k, v in enc.items()}
        enc["labels"] = torch.tensor(label, dtype=torch.long)
        return enc