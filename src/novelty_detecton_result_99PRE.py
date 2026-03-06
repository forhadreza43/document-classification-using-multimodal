import argparse
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import torch
from torch.utils.data import DataLoader

from config import Paths, TrainConfig
from data import RVLCDIPOCRTextDataset, RVLCDIPOODTextDataset, set_seed
from model import BertDocClassifier
from knn_ood import (
    extract_embeddings_and_logits,
    build_faiss_l2_index,
    knn1_score_and_neighbor,
)
from metrics import compute_end_to_end_metrics


LOSS_ORDER = ["margin", "margin_star", "scl", "weight", "ce"]


def format_loss_name(loss: str) -> str:
    return {
        "margin": "Margin",
        "margin_star": "Margin*",
        "scl": "SCL",
        "weight": "Weight",
        "ce": "CE",
    }.get(loss, loss)


def find_ckpts(ckpt_dir: Path, suffix: str) -> List[Tuple[str, Path]]:
    found = []
    for loss in LOSS_ORDER:
        p = ckpt_dir / f"bert_{loss}_{suffix}.pt"
        if p.exists():
            found.append((loss, p))
    return found


def compute_scores_preds_nns(train_index, embeddings: np.ndarray, logits: np.ndarray):
    """
    For each sample:
      - score = -euclidean_distance_to_1NN
      - pred = argmax(logits)
      - nn_label = label of 1NN
    """
    n = embeddings.shape[0]
    scores = np.zeros((n,), dtype=np.float32)
    pred = logits.argmax(axis=1).astype(np.int64)
    nn_labels = np.zeros((n,), dtype=np.int64)

    for i in range(n):
        s, _, nn_label = knn1_score_and_neighbor(train_index, embeddings[i])
        scores[i] = s
        nn_labels[i] = nn_label

    return scores, pred, nn_labels


def compute_pipeline_metrics_from_threshold(
    theta: float,
    y_true_id: np.ndarray,
    id_scores: np.ndarray,
    id_pred: np.ndarray,
    id_nn: np.ndarray,
    ood_scores: np.ndarray,
    ood_pred: np.ndarray,
    ood_nn: np.ndarray,
    use_agreement: bool,
) -> Dict[str, float]:
    """
    Computes PRE/REC/F1/COV at a given threshold.
    If use_agreement=True => KNN*
    Else => KNN
    """
    if use_agreement:
        accept_id = (id_scores >= theta) & (id_pred == id_nn)
        accept_ood = (ood_scores >= theta) & (ood_pred == ood_nn)
    else:
        accept_id = (id_scores >= theta)
        accept_ood = (ood_scores >= theta)

    tp = int(accept_id.sum())
    fn = int((~accept_id).sum())
    fp = int(accept_ood.sum())
    tn = int((~accept_ood).sum())

    tp_correct = int(((id_pred == y_true_id) & accept_id).sum())

    pre = tp_correct / max(1, tp + fp)
    rec = tp_correct / max(1, tp + fn)
    f1 = 0.0 if (pre + rec) == 0 else 2.0 * pre * rec / (pre + rec)
    cov = tp_correct / max(1, tp + fp + tn + fn)

    return {
        "pre": pre,
        "rec": rec,
        "f1": f1,
        "cov": cov,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "tp_correct": tp_correct,
    }


def select_theta_for_target_precision(
    target_precision: float,
    y_true_id: np.ndarray,
    id_scores: np.ndarray,
    id_pred: np.ndarray,
    id_nn: np.ndarray,
    ood_scores: np.ndarray,
    ood_pred: np.ndarray,
    ood_nn: np.ndarray,
    use_agreement: bool,
):
    """
    Select threshold theta such that PRE >= target_precision on calibration data.

    Selection rule:
      1) among all thresholds satisfying PRE >= target, choose the one with highest REC
      2) tie-break by highest COV
      3) tie-break by lowest theta (more permissive, more coverage)

    If no threshold satisfies target precision:
      choose threshold with highest PRE, then highest REC, then highest COV.

    Returns:
      best_theta, best_metrics, target_met (bool)
    """
    all_scores = np.concatenate([id_scores, ood_scores], axis=0)
    candidates = np.unique(all_scores)
    candidates = np.sort(candidates)  # ascending; lower theta accepts more
    # Also include one threshold slightly above max score to represent "reject almost all"
    candidates = np.concatenate([candidates, np.array([all_scores.max() + 1e-6], dtype=np.float32)])

    satisfying = []
    fallback = []

    for theta in candidates:
        m = compute_pipeline_metrics_from_threshold(
            theta=float(theta),
            y_true_id=y_true_id,
            id_scores=id_scores,
            id_pred=id_pred,
            id_nn=id_nn,
            ood_scores=ood_scores,
            ood_pred=ood_pred,
            ood_nn=ood_nn,
            use_agreement=use_agreement,
        )
        record = (float(theta), m)

        if m["pre"] >= target_precision:
            satisfying.append(record)
        fallback.append(record)

    if satisfying:
        # maximize REC, then COV, then choose lower theta
        satisfying.sort(key=lambda x: (x[1]["rec"], x[1]["cov"], -x[0]), reverse=True)
        best_theta, best_metrics = satisfying[0]
        return best_theta, best_metrics, True

    # No threshold reaches target precision
    fallback.sort(key=lambda x: (x[1]["pre"], x[1]["rec"], x[1]["cov"], -x[0]), reverse=True)
    best_theta, best_metrics = fallback[0]
    return best_theta, best_metrics, False


def apply_threshold_on_eval(
    theta: float,
    y_true_id: np.ndarray,
    id_scores: np.ndarray,
    id_pred: np.ndarray,
    id_nn: np.ndarray,
    ood_scores: np.ndarray,
    ood_pred: np.ndarray,
    ood_nn: np.ndarray,
    use_agreement: bool,
):
    """
    Build the exact arrays expected by compute_end_to_end_metrics() on evaluation data.
    """
    if use_agreement:
        is_id_pred_idset = (id_scores >= theta) & (id_pred == id_nn)
        is_id_pred_oodset = (ood_scores >= theta) & (ood_pred == ood_nn)
    else:
        is_id_pred_idset = (id_scores >= theta)
        is_id_pred_oodset = (ood_scores >= theta)

    y_pred_idset = np.full_like(y_true_id, fill_value=-1)
    y_pred_idset[is_id_pred_idset] = id_pred[is_id_pred_idset]

    return compute_end_to_end_metrics(
        y_true_id=y_true_id,
        is_id_pred_idset=is_id_pred_idset,
        y_pred_idset=y_pred_idset,
        is_id_pred_oodset=is_id_pred_oodset,
    )


def split_ood_for_calibration_and_eval(
    embeddings: np.ndarray,
    logits: np.ndarray,
    calib_frac: float,
    seed: int,
):
    """
    Splits OOD set into calibration and evaluation subsets.
    """
    n = embeddings.shape[0]
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)

    n_calib = max(1, int(round(n * calib_frac)))
    n_calib = min(n_calib, n - 1) if n > 1 else 1

    calib_idx = idx[:n_calib]
    eval_idx = idx[n_calib:] if n > 1 else idx[:1]

    return (
        embeddings[calib_idx],
        logits[calib_idx],
        embeddings[eval_idx],
        logits[eval_idx],
    )


def eval_one_checkpoint(project_root: Path, ckpt_path: Path, target_precision: float, ood_calib_frac: float):
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

    train_emb, train_logits, train_y = extract_embeddings_and_logits(model, train_loader, device=str(device))
    val_emb, val_logits, val_y = extract_embeddings_and_logits(model, val_loader, device=str(device))
    test_emb, test_logits, test_y = extract_embeddings_and_logits(model, test_loader, device=str(device))
    ood_emb, ood_logits, _ = extract_embeddings_and_logits(model, ood_loader, device=str(device))

    train_index = build_faiss_l2_index(train_emb.astype(np.float32), train_y.astype(np.int64))

    # Split OOD into calibration and evaluation subsets
    ood_calib_emb, ood_calib_logits, ood_eval_emb, ood_eval_logits = split_ood_for_calibration_and_eval(
        embeddings=ood_emb,
        logits=ood_logits,
        calib_frac=ood_calib_frac,
        seed=cfg.seed,
    )

    # Calibration features
    val_scores, val_pred, val_nn = compute_scores_preds_nns(train_index, val_emb.astype(np.float32), val_logits)
    ood_calib_scores, ood_calib_pred, ood_calib_nn = compute_scores_preds_nns(
        train_index, ood_calib_emb.astype(np.float32), ood_calib_logits
    )

    # Eval features
    test_scores, test_pred, test_nn = compute_scores_preds_nns(train_index, test_emb.astype(np.float32), test_logits)
    ood_eval_scores, ood_eval_pred, ood_eval_nn = compute_scores_preds_nns(
        train_index, ood_eval_emb.astype(np.float32), ood_eval_logits
    )

    # KNN threshold from PRE>=target
    theta_knn, calib_knn, met_knn = select_theta_for_target_precision(
        target_precision=target_precision,
        y_true_id=val_y,
        id_scores=val_scores,
        id_pred=val_pred,
        id_nn=val_nn,
        ood_scores=ood_calib_scores,
        ood_pred=ood_calib_pred,
        ood_nn=ood_calib_nn,
        use_agreement=False,
    )

    # KNN* threshold from PRE>=target
    theta_knn_star, calib_knn_star, met_knn_star = select_theta_for_target_precision(
        target_precision=target_precision,
        y_true_id=val_y,
        id_scores=val_scores,
        id_pred=val_pred,
        id_nn=val_nn,
        ood_scores=ood_calib_scores,
        ood_pred=ood_calib_pred,
        ood_nn=ood_calib_nn,
        use_agreement=True,
    )

    # Evaluate on held-out ID test + OOD eval split
    eval_knn = apply_threshold_on_eval(
        theta=theta_knn,
        y_true_id=test_y,
        id_scores=test_scores,
        id_pred=test_pred,
        id_nn=test_nn,
        ood_scores=ood_eval_scores,
        ood_pred=ood_eval_pred,
        ood_nn=ood_eval_nn,
        use_agreement=False,
    )

    eval_knn_star = apply_threshold_on_eval(
        theta=theta_knn_star,
        y_true_id=test_y,
        id_scores=test_scores,
        id_pred=test_pred,
        id_nn=test_nn,
        ood_scores=ood_eval_scores,
        ood_pred=ood_eval_pred,
        ood_nn=ood_eval_nn,
        use_agreement=True,
    )

    return {
        "knn": {
            "theta": theta_knn,
            "target_met": met_knn,
            "calib_pre": calib_knn["pre"] * 100.0,
            "pre": eval_knn.pre * 100.0,
            "rec": eval_knn.rec * 100.0,
            "f1": eval_knn.f1 * 100.0,
            "cov": eval_knn.cov * 100.0,
        },
        "knn_star": {
            "theta": theta_knn_star,
            "target_met": met_knn_star,
            "calib_pre": calib_knn_star["pre"] * 100.0,
            "pre": eval_knn_star.pre * 100.0,
            "rec": eval_knn_star.rec * 100.0,
            "f1": eval_knn_star.f1 * 100.0,
            "cov": eval_knn_star.cov * 100.0,
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_root", required=True)
    parser.add_argument("--ckpt_dir", default="checkpoints")
    parser.add_argument("--suffix", default="debug", choices=["debug", "full"])
    parser.add_argument("--target_pre", type=float, default=99.0, help="Target precision in percent, default 99")
    parser.add_argument("--ood_calib_frac", type=float, default=0.5, help="Fraction of OOD used for calibration")
    args = parser.parse_args()

    project_root = Path(args.project_root)
    ckpt_dir = Path(args.ckpt_dir)
    target_precision = args.target_pre / 100.0

    ckpts = find_ckpts(ckpt_dir, args.suffix)
    if not ckpts:
        print(f"No checkpoints found in {ckpt_dir} with suffix '{args.suffix}'.")
        print("Expected names like bert_margin_star_debug.pt, bert_ce_debug.pt, etc.")
        return

    results = []
    for loss, ckpt_path in ckpts:
        metrics = eval_one_checkpoint(
            project_root=project_root,
            ckpt_path=ckpt_path,
            target_precision=target_precision,
            ood_calib_frac=args.ood_calib_frac,
        )
        results.append((loss, metrics))

    results.sort(key=lambda x: LOSS_ORDER.index(x[0]))

    print(f"\nKNN results at target PRE ≥ {args.target_pre:.2f}%")
    print(f"{'#':<3} {'Loss':<10} {'PRE↑':>8} {'REC↑':>8} {'F1↑':>8} {'COV↑':>8} {'Met?':>6}")
    print("-" * 58)
    for i, (loss, m) in enumerate(results, start=1):
        r = m["knn"]
        met = "Y" if r["target_met"] else "N"
        print(f"{i:<3} {format_loss_name(loss):<10} {r['pre']:8.2f} {r['rec']:8.2f} {r['f1']:8.2f} {r['cov']:8.2f} {met:>6}")

    print(f"\nKNN* results at target PRE ≥ {args.target_pre:.2f}%")
    print(f"{'#':<3} {'Loss':<10} {'PRE↑':>8} {'REC↑':>8} {'F1↑':>8} {'COV↑':>8} {'Met?':>6}")
    print("-" * 58)
    for i, (loss, m) in enumerate(results, start=1):
        r = m["knn_star"]
        met = "Y" if r["target_met"] else "N"
        print(f"{i:<3} {format_loss_name(loss):<10} {r['pre']:8.2f} {r['rec']:8.2f} {r['f1']:8.2f} {r['cov']:8.2f} {met:>6}")

    print("\nNotes:")
    print("- Threshold θ is selected on calibration data to achieve target precision if possible.")
    print("- ID calibration set = RVL-CDIP validation split.")
    print("- OOD calibration set = random fraction of RVL-CDIP-O (controlled by --ood_calib_frac).")
    print("- Reported PRE/REC/F1/COV are measured on ID test + held-out OOD evaluation subset.")
    print("- 'Met?' indicates whether the calibration threshold actually reached the requested precision.")


if __name__ == "__main__":
    main()