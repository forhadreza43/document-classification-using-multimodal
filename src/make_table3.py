import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from config import Paths, TrainConfig, OODConfig
from data import RVLCDIPOCRTextDataset, RVLCDIPOODTextDataset, set_seed
from model import BertDocClassifier
from knn_ood import (
    extract_embeddings_and_logits,
    build_faiss_l2_index,
    estimate_threshold_theta,
    knn_star_predict,
    knn_predict_no_agreement,
    knn1_score_and_neighbor,
)
from metrics import compute_auc, compute_fpr_at_tpr95, compute_end_to_end_metrics


LOSS_ORDER = ["margin", "margin_star", "scl", "weight", "ce"]


def compute_scores(train_index, embeddings: np.ndarray) -> np.ndarray:
    scores = np.zeros((embeddings.shape[0],), dtype=np.float32)
    for i in range(embeddings.shape[0]):
        s, _, _ = knn1_score_and_neighbor(train_index, embeddings[i])
        scores[i] = s
    return scores


def find_ckpts(ckpt_dir: Path, suffix: str) -> List[Tuple[str, Path]]:
    found = []
    for loss in LOSS_ORDER:
        p = ckpt_dir / f"bert_{loss}_{suffix}.pt"
        if p.exists():
            found.append((loss, p))
    return found


def format_loss_name(loss: str) -> str:
    return {
        "margin": "Margin",
        "margin_star": "Margin*",
        "scl": "SCL",
        "weight": "Weight",
        "ce": "CE",
    }.get(loss, loss)


def eval_one_checkpoint(project_root: Path, ckpt_path: Path) -> dict:
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
    ood_cfg = OODConfig()
    set_seed(cfg.seed)
    device = torch.device(cfg.device)

    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    model_name = ckpt["model_name"]
    max_length = ckpt["max_length"]
    num_classes = int(ckpt.get("num_classes", 16))

    model = BertDocClassifier(model_name, num_classes=num_classes).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    is_debug = ckpt_path.stem.endswith("_debug")
    debug_samples = cfg.debug_samples if is_debug else None

    train_ds = RVLCDIPOCRTextDataset(
        qs_root=paths.qs_ocr_large_dir,
        split_file=paths.train_list,
        tokenizer_name=model_name,
        max_length=max_length,
        debug_samples=debug_samples,
    )
    val_ds = RVLCDIPOCRTextDataset(
        qs_root=paths.qs_ocr_large_dir,
        split_file=paths.val_list,
        tokenizer_name=model_name,
        max_length=max_length,
        debug_samples=debug_samples,
    )
    test_ds = RVLCDIPOCRTextDataset(
        qs_root=paths.qs_ocr_large_dir,
        split_file=paths.test_list,
        tokenizer_name=model_name,
        max_length=max_length,
        debug_samples=debug_samples,
    )
    ood_ds = RVLCDIPOODTextDataset(
        ood_text_dir=paths.rvl_cdip_ood_text_dir,
        tokenizer_name=model_name,
        max_length=max_length,
        debug_samples=debug_samples,
    )

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=False, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False, num_workers=0)
    ood_loader = DataLoader(ood_ds, batch_size=16, shuffle=False, num_workers=0)

    train_emb, _, train_y = extract_embeddings_and_logits(model, train_loader, device=str(device))
    val_emb, _, _ = extract_embeddings_and_logits(model, val_loader, device=str(device))
    test_emb, test_logits, test_y = extract_embeddings_and_logits(model, test_loader, device=str(device))
    ood_emb, ood_logits, _ = extract_embeddings_and_logits(model, ood_loader, device=str(device))

    train_index = build_faiss_l2_index(train_emb.astype(np.float32), train_y.astype(np.int64))

    theta = estimate_threshold_theta(
        train_index=train_index,
        val_embeddings=val_emb.astype(np.float32),
        tpr_target=ood_cfg.tpr_target,
    )

    id_scores = compute_scores(train_index, test_emb.astype(np.float32))
    ood_scores = compute_scores(train_index, ood_emb.astype(np.float32))

    auc = compute_auc(id_scores, ood_scores) * 100.0
    fpr = compute_fpr_at_tpr95(id_scores, ood_scores, tpr_target=ood_cfg.tpr_target) * 100.0

    is_id_test_knn, pred_test_knn = knn_predict_no_agreement(train_index, test_emb, test_logits, theta)
    is_id_ood_knn, _ = knn_predict_no_agreement(train_index, ood_emb, ood_logits, theta)
    knn_metrics = compute_end_to_end_metrics(
        y_true_id=test_y,
        is_id_pred_idset=is_id_test_knn,
        y_pred_idset=pred_test_knn,
        is_id_pred_oodset=is_id_ood_knn,
    )

    is_id_test_knn_star, pred_test_knn_star = knn_star_predict(train_index, test_emb, test_logits, theta)
    is_id_ood_knn_star, _ = knn_star_predict(train_index, ood_emb, ood_logits, theta)
    knn_star_metrics = compute_end_to_end_metrics(
        y_true_id=test_y,
        is_id_pred_idset=is_id_test_knn_star,
        y_pred_idset=pred_test_knn_star,
        is_id_pred_oodset=is_id_ood_knn_star,
    )

    return {
        "fpr": fpr,
        "auc": auc,
        "knn_f1": knn_metrics.f1 * 100.0,
        "knn_cov": knn_metrics.cov * 100.0,
        "knn_star_f1": knn_star_metrics.f1 * 100.0,
        "knn_star_cov": knn_star_metrics.cov * 100.0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_root", required=True)
    parser.add_argument("--ckpt_dir", default="checkpoints")
    parser.add_argument("--suffix", default="debug", choices=["debug", "full"])
    args = parser.parse_args()

    project_root = Path(args.project_root)
    ckpt_dir = Path(args.ckpt_dir)

    ckpts = find_ckpts(ckpt_dir, args.suffix)
    if not ckpts:
        print(f"No checkpoints found in {ckpt_dir} with suffix '{args.suffix}'.")
        return

    results = []
    for loss, ckpt_path in ckpts:
        metrics = eval_one_checkpoint(project_root, ckpt_path)
        results.append((loss, metrics))

    results.sort(key=lambda x: LOSS_ORDER.index(x[0]))

    print("\nKNN results (RVL-CDIP)")
    print(f"{'#':<3} {'Loss':<10} {'FPR↓':>8} {'AUC↑':>8} {'F1↑':>8} {'COV↑':>8}")
    print("-" * 45)
    for i, (loss, m) in enumerate(results, start=1):
        print(f"{i:<3} {format_loss_name(loss):<10} {m['fpr']:8.2f} {m['auc']:8.2f} {m['knn_f1']:8.2f} {m['knn_cov']:8.2f}")

    print("\nKNN* results (RVL-CDIP)")
    print(f"{'#':<3} {'Loss':<10} {'FPR↓':>8} {'AUC↑':>8} {'F1↑':>8} {'COV↑':>8}")
    print("-" * 45)
    for i, (loss, m) in enumerate(results, start=1):
        print(f"{i:<3} {format_loss_name(loss):<10} {m['fpr']:8.2f} {m['auc']:8.2f} {m['knn_star_f1']:8.2f} {m['knn_star_cov']:8.2f}")

    print("\nNotes:")
    print("- FPR is FPR@TPR95.")
    print("- AUC is computed from ID test vs OOD scores.")
    print("- KNN* uses consensus agreement for ambiguity rejection.")


if __name__ == "__main__":
    main()