import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score

from config import Paths, TrainConfig
from data import RVLCDIPOCRTextDataset, set_seed
from model import BertDocClassifier
from knn_ood import extract_embeddings_and_logits


LOSS_ORDER = ["margin_star", "margin", "scl", "weight", "ce"]


def eval_closed_set(project_root: Path, ckpt_path: Path, batch_size: int = 16) -> Tuple[float, float, float]:
    """
    Returns (ACC, wPRE, mREC) as percentages (0-100).
    """
    paths = Paths(
        project_root=project_root,
        qs_ocr_large_dir=project_root / "QS-OCR-Large",
        rvl_cdip_dir=project_root / "rvl-cdip",
        rvl_cdip_ood_text_dir=project_root / "rvl-cdip-o-text",
        train_list=project_root / "train.txt",
        val_list=project_root / "val.txt",
        test_list=project_root / "test.txt",
    )

    cfg = TrainConfig()
    set_seed(cfg.seed)
    device = torch.device(cfg.device)

    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    model_name = ckpt["model_name"]
    max_length = ckpt["max_length"]
    num_classes = int(ckpt.get("num_classes", 16))

    model = BertDocClassifier(model_name, num_classes=num_classes).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # Closed-set test on RVL-CDIP test.txt
    test_ds = RVLCDIPOCRTextDataset(
        qs_root=paths.qs_ocr_large_dir,
        split_file=paths.test_list,
        tokenizer_name=model_name,
        max_length=max_length,
        debug_samples=cfg.debug_samples,
    )
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    _, logits, y = extract_embeddings_and_logits(model, test_loader, device=str(device))
    y_true = y.astype(np.int64)
    y_pred = logits.argmax(axis=1).astype(np.int64)

    acc = accuracy_score(y_true, y_pred) * 100.0
    wpre = precision_score(y_true, y_pred, average="weighted", zero_division=0) * 100.0
    mrec = recall_score(y_true, y_pred, average="macro", zero_division=0) * 100.0
    return acc, wpre, mrec


def find_ckpts(ckpt_dir: Path, suffix: str) -> List[Tuple[str, Path]]:
    """
    Auto-detect checkpoints:
      checkpoints/bert_<loss>_<suffix>.pt
    where suffix is 'debug' or 'full'.
    """
    found = []
    for loss in LOSS_ORDER:
        p = ckpt_dir / f"bert_{loss}_{suffix}.pt"
        if p.exists():
            found.append((loss, p))
    return found


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project_root", type=str, required=True)
    ap.add_argument("--ckpt_dir", type=str, default="checkpoints")
    ap.add_argument("--suffix", type=str, default="debug", choices=["debug", "full"])
    ap.add_argument("--batch_size", type=int, default=16)
    args = ap.parse_args()

    project_root = Path(args.project_root)
    ckpt_dir = Path(args.ckpt_dir)
    suffix = args.suffix

    ckpts = find_ckpts(ckpt_dir, suffix=suffix)
    if not ckpts:
        print(f"No checkpoints found in {ckpt_dir} with suffix '{suffix}'.")
        print("Expected names like: bert_margin_star_debug.pt, bert_ce_debug.pt, ...")
        return

    rows = []
    for loss, ckpt_path in ckpts:
        acc, wpre, mrec = eval_closed_set(project_root, ckpt_path, batch_size=args.batch_size)
        rows.append((loss, acc, wpre, mrec))

    # Keep paper-like ordering
    rows.sort(key=lambda x: LOSS_ORDER.index(x[0]) if x[0] in LOSS_ORDER else 999)

    print("\nClassification results (RVL-CDIP test set)")
    print(f"{'#':<3} {'Loss':<12} {'ACC↑':>8} {'wPRE↑':>8} {'mREC↑':>8}")
    print("-" * 45)
    for i, (loss, acc, wpre, mrec) in enumerate(rows, start=1):
        print(f"{i:<3} {loss:<12} {acc:8.2f} {wpre:8.2f} {mrec:8.2f}")

    print("\nNotes:")
    print("- ACC  = accuracy")
    print("- wPRE = weighted precision (sklearn average='weighted')")
    print("- mREC = mean recall = macro recall (sklearn average='macro')")
    print(f"- Loaded checkpoints from: {ckpt_dir.resolve()}")
    print(f"- Using suffix: {suffix}  (choose --suffix full after full training)")


if __name__ == "__main__":
    main()