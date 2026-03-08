import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score

from config import Paths, TrainConfig
from data import RVLCDIPOCRTextDataset, set_seed
from data_multimodal import RVLCDIPLayoutLMv3Dataset
from model import BertDocClassifier
from model_multimodal import LayoutLMv3DocClassifier
from knn_ood import extract_embeddings_and_logits
from knn_ood_multimodal import extract_embeddings_and_logits_multimodal


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project_root", required=True)
    ap.add_argument("--text_ckpt", required=True)
    ap.add_argument("--multi_ckpt", required=True)
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

    # ---------- Text-only ----------
    text_ckpt = torch.load(args.text_ckpt, map_location="cpu")
    text_model = BertDocClassifier(
        text_ckpt["model_name"],
        num_classes=text_ckpt.get("num_classes", 16)
    ).to(device)
    text_model.load_state_dict(text_ckpt["model_state"])
    text_model.eval()

    text_test_ds = RVLCDIPOCRTextDataset(
        qs_root=paths.qs_ocr_large_dir,
        split_file=paths.test_list,
        tokenizer_name=text_ckpt["model_name"],
        max_length=text_ckpt["max_length"],
        debug_samples=cfg.debug_samples if Path(args.text_ckpt).stem.endswith("_debug") else None,
    )
    text_test_loader = DataLoader(text_test_ds, batch_size=16, shuffle=False, num_workers=0)

    _, text_logits, text_y = extract_embeddings_and_logits(text_model, text_test_loader, device=str(device))
    text_pred = text_logits.argmax(axis=1)

    text_acc = accuracy_score(text_y, text_pred) * 100.0
    text_wpre = precision_score(text_y, text_pred, average="weighted", zero_division=0) * 100.0
    text_mrec = recall_score(text_y, text_pred, average="macro", zero_division=0) * 100.0

    # ---------- Multimodal ----------
    multi_ckpt = torch.load(args.multi_ckpt, map_location="cpu")
    multi_model = LayoutLMv3DocClassifier(
        multi_ckpt["model_name"],
        num_classes=multi_ckpt.get("num_classes", 16)
    ).to(device)
    multi_model.load_state_dict(multi_ckpt["model_state"])
    multi_model.eval()

    multi_test_ds = RVLCDIPLayoutLMv3Dataset(
        rvl_root=paths.rvl_cdip_dir,
        ocr_box_root=ocr_box_root,
        split_file=paths.test_list,
        processor_name=multi_ckpt["model_name"],
        max_length=multi_ckpt["max_length"],
        debug_samples=cfg.debug_samples if Path(args.multi_ckpt).stem.endswith("_debug") else None,
    )
    multi_test_loader = DataLoader(multi_test_ds, batch_size=8, shuffle=False, num_workers=0)

    _, multi_logits, multi_y = extract_embeddings_and_logits_multimodal(multi_model, multi_test_loader, device=str(device))
    multi_pred = multi_logits.argmax(axis=1)

    multi_acc = accuracy_score(multi_y, multi_pred) * 100.0
    multi_wpre = precision_score(multi_y, multi_pred, average="weighted", zero_division=0) * 100.0
    multi_mrec = recall_score(multi_y, multi_pred, average="macro", zero_division=0) * 100.0

    print("\nPhase 3 — Closed-set classification comparison")
    print(f"{'#':<3} {'Model':<18} {'ACC↑':>8} {'wPRE↑':>8} {'mREC↑':>8}")
    print("-" * 50)
    print(f"{1:<3} {'Text-BERT':<18} {text_acc:8.2f} {text_wpre:8.2f} {text_mrec:8.2f}")
    print(f"{2:<3} {'LayoutLMv3':<18} {multi_acc:8.2f} {multi_wpre:8.2f} {multi_mrec:8.2f}")


if __name__ == "__main__":
    main()