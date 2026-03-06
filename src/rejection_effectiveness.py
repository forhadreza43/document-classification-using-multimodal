import argparse
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

from config import Paths, TrainConfig
from data import RVLCDIPOCRTextDataset, RVLCDIPOODTextDataset, set_seed
from model import BertDocClassifier
from knn_ood import (
    extract_embeddings_and_logits,
    build_faiss_l2_index,
    knn1_score_and_neighbor
)
from metrics import compute_end_to_end_metrics


LOSS_ORDER = ["margin", "margin_star", "scl", "weight", "ce"]


def find_ckpts(ckpt_dir, suffix):
    ckpts = []
    for loss in LOSS_ORDER:
        p = ckpt_dir / f"bert_{loss}_{suffix}.pt"
        if p.exists():
            ckpts.append((loss, p))
    return ckpts


def format_loss(loss):
    return {
        "margin": "Margin",
        "margin_star": "Margin*",
        "scl": "SCL",
        "weight": "Weight",
        "ce": "CE"
    }.get(loss, loss)


def compute_scores_preds_nns(train_index, emb, logits):

    n = emb.shape[0]

    scores = np.zeros(n)
    pred = logits.argmax(axis=1)
    nn_labels = np.zeros(n)

    for i in range(n):

        s, _, nn = knn1_score_and_neighbor(train_index, emb[i])

        scores[i] = s
        nn_labels[i] = nn

    return scores, pred, nn_labels


def compute_metrics(theta, y_true, scores, pred, nn, ood_scores, ood_pred, ood_nn):

    accept_id = (scores >= theta) & (pred == nn)
    accept_ood = (ood_scores >= theta) & (ood_pred == ood_nn)

    y_pred_id = np.full_like(y_true, -1)
    y_pred_id[accept_id] = pred[accept_id]

    return compute_end_to_end_metrics(
        y_true_id=y_true,
        is_id_pred_idset=accept_id,
        y_pred_idset=y_pred_id,
        is_id_pred_oodset=accept_ood
    )


def find_theta(target_precision, y_true, scores, pred, nn, ood_scores, ood_pred, ood_nn):

    thresholds = np.unique(np.concatenate([scores, ood_scores]))

    best = None

    for t in thresholds:

        m = compute_metrics(t, y_true, scores, pred, nn, ood_scores, ood_pred, ood_nn)

        if m.pre >= target_precision:

            if best is None or m.rec > best[1].rec:
                best = (t, m)

    if best is None:

        best = (thresholds.max(), compute_metrics(
            thresholds.max(), y_true, scores, pred, nn,
            ood_scores, ood_pred, ood_nn
        ))

    return best


def evaluate_loss(project_root, ckpt_path, target_pre):

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

    ckpt = torch.load(ckpt_path, map_location="cpu")

    model = BertDocClassifier(
        ckpt["model_name"],
        num_classes=ckpt.get("num_classes", 16)
    ).to(device)

    model.load_state_dict(ckpt["model_state"])
    model.eval()

    max_length = ckpt["max_length"]

    val_ds = RVLCDIPOCRTextDataset(
        qs_root=paths.qs_ocr_large_dir,
        split_file=paths.val_list,
        tokenizer_name=ckpt["model_name"],
        max_length=max_length,
        debug_samples=cfg.debug_samples
    )

    test_ds = RVLCDIPOCRTextDataset(
        qs_root=paths.qs_ocr_large_dir,
        split_file=paths.test_list,
        tokenizer_name=ckpt["model_name"],
        max_length=max_length,
        debug_samples=cfg.debug_samples
    )

    ood_ds = RVLCDIPOODTextDataset(
        ood_text_dir=paths.rvl_cdip_ood_text_dir,
        tokenizer_name=ckpt["model_name"],
        max_length=max_length,
        debug_samples=cfg.debug_samples
    )

    train_ds = RVLCDIPOCRTextDataset(
        qs_root=paths.qs_ocr_large_dir,
        split_file=paths.train_list,
        tokenizer_name=ckpt["model_name"],
        max_length=max_length,
        debug_samples=cfg.debug_samples
    )

    train_loader = DataLoader(train_ds, batch_size=16)
    val_loader = DataLoader(val_ds, batch_size=16)
    test_loader = DataLoader(test_ds, batch_size=16)
    ood_loader = DataLoader(ood_ds, batch_size=16)

    train_emb, _, train_y = extract_embeddings_and_logits(model, train_loader, device=str(device))
    val_emb, val_logits, val_y = extract_embeddings_and_logits(model, val_loader, device=str(device))
    test_emb, test_logits, test_y = extract_embeddings_and_logits(model, test_loader, device=str(device))
    ood_emb, ood_logits, _ = extract_embeddings_and_logits(model, ood_loader, device=str(device))

    index = build_faiss_l2_index(train_emb.astype(np.float32), train_y.astype(np.int64))

    val_scores, val_pred, val_nn = compute_scores_preds_nns(index, val_emb, val_logits)
    ood_scores, ood_pred, ood_nn = compute_scores_preds_nns(index, ood_emb, ood_logits)

    theta, _ = find_theta(target_pre, val_y, val_scores, val_pred, val_nn, ood_scores, ood_pred, ood_nn)

    test_scores, test_pred, test_nn = compute_scores_preds_nns(index, test_emb, test_logits)

    metrics = compute_metrics(theta, test_y, test_scores, test_pred, test_nn, ood_scores, ood_pred, ood_nn)

    return metrics


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--project_root", required=True)
    parser.add_argument("--ckpt_dir", default="checkpoints")
    parser.add_argument("--suffix", default="debug")

    args = parser.parse_args()

    project_root = Path(args.project_root)
    ckpt_dir = Path(args.ckpt_dir)

    thresholds = [0.99, 0.988, 0.98]

    ckpts = find_ckpts(ckpt_dir, args.suffix)

    for thr in thresholds:

        print(f"\nThreshold = {thr*100:.1f}%\n")

        print(f"{'#':<3} {'Loss':<10} {'PRE↑':>8} {'REC↑':>8} {'F1↑':>8} {'COV↑':>8}")
        print("-"*45)

        for i,(loss,ckpt) in enumerate(ckpts,1):

            m = evaluate_loss(project_root, ckpt, thr)

            print(f"{i:<3} {format_loss(loss):<10} {m.pre*100:8.2f} {m.rec*100:8.2f} {m.f1*100:8.2f} {m.cov*100:8.2f}")


if __name__ == "__main__":
    main()