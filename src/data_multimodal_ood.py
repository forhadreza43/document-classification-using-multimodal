from pathlib import Path
from typing import Optional, Dict, List

import torch
from torch.utils.data import Dataset
from PIL import Image
from transformers import AutoProcessor

from data_multimodal import load_words_and_boxes, normalize_box


class RVLCDIPOODLayoutLMv3Dataset(Dataset):
    def __init__(
        self,
        rvl_ood_root: Path,
        ocr_box_root: Path,
        split_dir: Path,
        processor_name: str = "microsoft/layoutlmv3-base",
        max_length: int = 512,
        debug_samples: Optional[int] = None,
    ):
        self.items = []
        txt_files = sorted(split_dir.glob("*.txt"))
        if debug_samples is not None:
            txt_files = txt_files[:debug_samples]

        for txt_file in txt_files:
            stem = txt_file.stem
            img_path = rvl_ood_root / f"{stem}.tif"
            tsv_path = ocr_box_root / f"{stem}.tsv"
            if img_path.exists() and tsv_path.exists():
                self.items.append((img_path, tsv_path))

        self.processor = AutoProcessor.from_pretrained(processor_name, apply_ocr=False)
        self.max_length = max_length

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        img_path, tsv_path = self.items[idx]
        image = Image.open(img_path).convert("RGB")
        width, height = image.size

        words, boxes_px = load_words_and_boxes(tsv_path)
        boxes = [normalize_box(b, width, height) for b in boxes_px]

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
        enc["labels"] = torch.tensor(-1, dtype=torch.long)
        return enc