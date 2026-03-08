import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score

from config import Paths, TrainConfig
from data import set_seed
from data_multimodal import RVLCDIPLayoutLMv3Dataset
from model_multimodal import LayoutLMv3DocClassifier


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project_root", type=str, required=True)
    ap.add_argument("--ckpt", type=str, required=True)
    args = ap.parse_args()

    cfg = TrainConfig()
    set_seed(cfg.seed)
    device = torch.device(cfg.device)

    project_root = Path(args.project_root)
    paths = Paths(
        project_root=project_root,
        qs_ocr_large_dir=project_root / "QS-OCR-Large",
        rvl_cdip_dir=project_root / "rvl-cdip",
        rvl_cdip_ood_text_dir=project_root / "rvl-cdip-o-text",
        train_list=project_root / "train.txt",
        val_list=project_root / "val.txt",
        test_list=project_root / "test.txt",
    )
    ocr_box_root = project_root / "QS-OCR-BOX"

    ckpt = torch.load(args.ckpt, map_location="cpu")

    model = LayoutLMv3DocClassifier(
        ckpt["model_name"],
        num_classes=ckpt.get("num_classes", 16)
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    test_ds = RVLCDIPLayoutLMv3Dataset(
        rvl_root=paths.rvl_cdip_dir,
        ocr_box_root=ocr_box_root,
        split_file=paths.test_list,
        processor_name=ckpt["model_name"],
        max_length=ckpt["max_length"],
        debug_samples=cfg.debug_samples,
    )
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)

    y_true = []
    y_pred = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            bbox = batch["bbox"].to(device)
            pixel_values = batch["pixel_values"].to(device)
            labels_t = batch["labels"].cpu().numpy()

            logits, _ = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                bbox=bbox,
                pixel_values=pixel_values,
            )
            pred = logits.argmax(dim=1).cpu().numpy()

            y_true.append(labels_t)
            y_pred.append(pred)

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)

    acc = accuracy_score(y_true, y_pred) * 100.0
    wpre = precision_score(y_true, y_pred, average="weighted", zero_division=0) * 100.0
    mrec = recall_score(y_true, y_pred, average="macro", zero_division=0) * 100.0

    print("\nPhase 1 multimodal closed-set results")
    print(f"ACC : {acc:.2f}")
    print(f"wPRE: {wpre:.2f}")
    print(f"mREC: {mrec:.2f}")


if __name__ == "__main__":
    main()