import argparse
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

from config import Paths, TrainConfig
from data import set_seed
from data_multimodal import RVLCDIPLayoutLMv3Dataset
from data_multimodal_ood import RVLCDIPOODLayoutLMv3Dataset
from model_multimodal import LayoutLMv3DocClassifier
from knn_ood_multimodal import (
    extract_embeddings_and_logits_multimodal,
    build_faiss_l2_index,
    knn1_score_and_neighbor,
)
from metrics import compute_end_to_end_metrics


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
        is_id_pred_oodset=accept_ood,
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
            thresholds.max(), y_true, scores, pred, nn, ood_scores, ood_pred, ood_nn
        ))
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project_root", required=True)
    ap.add_argument("--ckpt", required=True)
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
    ood_box_root = project_root / "QS-OCR-BOX-OOD"

    ckpt = torch.load(args.ckpt, map_location="cpu")
    is_debug = Path(args.ckpt).stem.endswith("_debug")
    debug_samples = cfg.debug_samples if is_debug else None

    model = LayoutLMv3DocClassifier(
        ckpt["model_name"],
        num_classes=ckpt.get("num_classes", 16)
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    train_ds = RVLCDIPLayoutLMv3Dataset(paths.rvl_cdip_dir, ocr_box_root, paths.train_list, ckpt["model_name"], ckpt["max_length"], debug_samples)
    val_ds = RVLCDIPLayoutLMv3Dataset(paths.rvl_cdip_dir, ocr_box_root, paths.val_list, ckpt["model_name"], ckpt["max_length"], debug_samples)
    test_ds = RVLCDIPLayoutLMv3Dataset(paths.rvl_cdip_dir, ocr_box_root, paths.test_list, ckpt["model_name"], ckpt["max_length"], debug_samples)
    ood_ds = RVLCDIPOODLayoutLMv3Dataset(project_root / "rvl-cdip-o", ood_box_root, paths.rvl_cdip_ood_text_dir, ckpt["model_name"], ckpt["max_length"], debug_samples)

    train_loader = DataLoader(train_ds, batch_size=8, shuffle=False, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=8, shuffle=False, num_workers=0)
    ood_loader = DataLoader(ood_ds, batch_size=8, shuffle=False, num_workers=0)

    train_emb, _, train_y = extract_embeddings_and_logits_multimodal(model, train_loader, device=str(device))
    val_emb, val_logits, val_y = extract_embeddings_and_logits_multimodal(model, val_loader, device=str(device))
    test_emb, test_logits, test_y = extract_embeddings_and_logits_multimodal(model, test_loader, device=str(device))
    ood_emb, ood_logits, _ = extract_embeddings_and_logits_multimodal(model, ood_loader, device=str(device))

    index = build_faiss_l2_index(train_emb.astype(np.float32), train_y.astype(np.int64))

    val_scores, val_pred, val_nn = compute_scores_preds_nns(index, val_emb, val_logits)
    ood_scores, ood_pred, ood_nn = compute_scores_preds_nns(index, ood_emb, ood_logits)

    test_scores, test_pred, test_nn = compute_scores_preds_nns(index, test_emb, test_logits)

    thresholds = [0.99, 0.988, 0.98]

    print("\nPhase 3 — Multimodal LayoutLMv3 KNN* precision-threshold table")
    for thr in thresholds:
        theta, _ = find_theta(thr, val_y, val_scores, val_pred, val_nn, ood_scores, ood_pred, ood_nn)
        m = compute_metrics(theta, test_y, test_scores, test_pred, test_nn, ood_scores, ood_pred, ood_nn)

        print(f"\nThreshold = {thr*100:.1f}%")
        print(f"{'PRE↑':>8} {'REC↑':>8} {'F1↑':>8} {'COV↑':>8}")
        print("-" * 36)
        print(f"{m.pre*100:8.2f} {m.rec*100:8.2f} {m.f1*100:8.2f} {m.cov*100:8.2f}")


if __name__ == "__main__":
    main()