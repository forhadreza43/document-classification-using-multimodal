import argparse
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

from config import Paths, TrainConfig, OODConfig
from data import RVLCDIPOCRTextDataset, RVLCDIPOODTextDataset, set_seed
from data_multimodal import RVLCDIPLayoutLMv3Dataset
from data_multimodal_ood import RVLCDIPOODLayoutLMv3Dataset
from model import BertDocClassifier
from model_multimodal import LayoutLMv3DocClassifier

from knn_ood import (
    extract_embeddings_and_logits,
    build_faiss_l2_index,
    estimate_threshold_theta,
    knn_predict_no_agreement,
    knn_star_predict,
    knn1_score_and_neighbor,
)
from knn_ood_multimodal import (
    extract_embeddings_and_logits_multimodal,
    build_faiss_l2_index as build_faiss_l2_index_multi,
    estimate_threshold_theta as estimate_threshold_theta_multi,
    knn_predict_no_agreement as knn_predict_no_agreement_multi,
    knn_star_predict as knn_star_predict_multi,
    knn1_score_and_neighbor as knn1_score_and_neighbor_multi,
)
from metrics import compute_auc, compute_fpr_at_tpr95, compute_end_to_end_metrics


def compute_scores_text(train_index, embeddings):
    scores = np.zeros((embeddings.shape[0],), dtype=np.float32)
    for i in range(embeddings.shape[0]):
        s, _, _ = knn1_score_and_neighbor(train_index, embeddings[i])
        scores[i] = s
    return scores


def compute_scores_multi(train_index, embeddings):
    scores = np.zeros((embeddings.shape[0],), dtype=np.float32)
    for i in range(embeddings.shape[0]):
        s, _, _ = knn1_score_and_neighbor_multi(train_index, embeddings[i])
        scores[i] = s
    return scores


def evaluate_text(project_root: Path, ckpt_path: str, use_knn_star: bool):
    cfg = TrainConfig()
    ood_cfg = OODConfig()
    device = torch.device(cfg.device)

    paths = Paths(
        project_root=project_root,
        qs_ocr_large_dir=project_root / "QS-OCR-Large",
        rvl_cdip_dir=project_root / "rvl-cdip",
        rvl_cdip_ood_text_dir=project_root / "rvl-cdip-o-text",
        train_list=project_root / "train.txt",
        val_list=project_root / "val.txt",
        test_list=project_root / "test.txt",
    )

    ckpt = torch.load(ckpt_path, map_location="cpu")
    debug_samples = cfg.debug_samples if Path(ckpt_path).stem.endswith("_debug") else None

    model = BertDocClassifier(
        ckpt["model_name"],
        num_classes=ckpt.get("num_classes", 16)
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    train_ds = RVLCDIPOCRTextDataset(paths.qs_ocr_large_dir, paths.train_list, ckpt["model_name"], ckpt["max_length"], debug_samples)
    val_ds = RVLCDIPOCRTextDataset(paths.qs_ocr_large_dir, paths.val_list, ckpt["model_name"], ckpt["max_length"], debug_samples)
    test_ds = RVLCDIPOCRTextDataset(paths.qs_ocr_large_dir, paths.test_list, ckpt["model_name"], ckpt["max_length"], debug_samples)
    ood_ds = RVLCDIPOODTextDataset(paths.rvl_cdip_ood_text_dir, ckpt["model_name"], ckpt["max_length"], debug_samples)

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=False, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False, num_workers=0)
    ood_loader = DataLoader(ood_ds, batch_size=16, shuffle=False, num_workers=0)

    train_emb, _, train_y = extract_embeddings_and_logits(model, train_loader, device=str(device))
    val_emb, _, _ = extract_embeddings_and_logits(model, val_loader, device=str(device))
    test_emb, test_logits, test_y = extract_embeddings_and_logits(model, test_loader, device=str(device))
    ood_emb, ood_logits, _ = extract_embeddings_and_logits(model, ood_loader, device=str(device))

    train_index = build_faiss_l2_index(train_emb.astype(np.float32), train_y.astype(np.int64))
    theta = estimate_threshold_theta(train_index, val_emb.astype(np.float32), ood_cfg.tpr_target)

    id_scores = compute_scores_text(train_index, test_emb)
    ood_scores = compute_scores_text(train_index, ood_emb)

    auc = compute_auc(id_scores, ood_scores) * 100.0
    fpr = compute_fpr_at_tpr95(id_scores, ood_scores, ood_cfg.tpr_target) * 100.0

    if use_knn_star:
        is_id_test, pred_test = knn_star_predict(train_index, test_emb, test_logits, theta)
        is_id_ood, _ = knn_star_predict(train_index, ood_emb, ood_logits, theta)
    else:
        is_id_test, pred_test = knn_predict_no_agreement(train_index, test_emb, test_logits, theta)
        is_id_ood, _ = knn_predict_no_agreement(train_index, ood_emb, ood_logits, theta)

    e2e = compute_end_to_end_metrics(test_y, is_id_test, pred_test, is_id_ood)
    return auc, fpr, e2e.f1 * 100.0, e2e.cov * 100.0


def evaluate_multi(project_root: Path, ckpt_path: str, use_knn_star: bool):
    cfg = TrainConfig()
    ood_cfg = OODConfig()
    device = torch.device(cfg.device)

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

    ckpt = torch.load(ckpt_path, map_location="cpu")
    debug_samples = cfg.debug_samples if Path(ckpt_path).stem.endswith("_debug") else None

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
    val_emb, _, _ = extract_embeddings_and_logits_multimodal(model, val_loader, device=str(device))
    test_emb, test_logits, test_y = extract_embeddings_and_logits_multimodal(model, test_loader, device=str(device))
    ood_emb, ood_logits, _ = extract_embeddings_and_logits_multimodal(model, ood_loader, device=str(device))

    train_index = build_faiss_l2_index_multi(train_emb.astype(np.float32), train_y.astype(np.int64))
    theta = estimate_threshold_theta_multi(train_index, val_emb.astype(np.float32), ood_cfg.tpr_target)

    id_scores = compute_scores_multi(train_index, test_emb)
    ood_scores = compute_scores_multi(train_index, ood_emb)

    auc = compute_auc(id_scores, ood_scores) * 100.0
    fpr = compute_fpr_at_tpr95(id_scores, ood_scores, ood_cfg.tpr_target) * 100.0

    if use_knn_star:
        is_id_test, pred_test = knn_star_predict_multi(train_index, test_emb, test_logits, theta)
        is_id_ood, _ = knn_star_predict_multi(train_index, ood_emb, ood_logits, theta)
    else:
        is_id_test, pred_test = knn_predict_no_agreement_multi(train_index, test_emb, test_logits, theta)
        is_id_ood, _ = knn_predict_no_agreement_multi(train_index, ood_emb, ood_logits, theta)

    e2e = compute_end_to_end_metrics(test_y, is_id_test, pred_test, is_id_ood)
    return auc, fpr, e2e.f1 * 100.0, e2e.cov * 100.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project_root", required=True)
    ap.add_argument("--text_ckpt", required=True)
    ap.add_argument("--multi_ckpt", required=True)
    args = ap.parse_args()

    cfg = TrainConfig()
    set_seed(cfg.seed)

    project_root = Path(args.project_root)

    print("\nPhase 3 — OOD comparison (KNN)")
    print(f"{'#':<3} {'Model':<18} {'FPR↓':>8} {'AUC↑':>8} {'F1↑':>8} {'COV↑':>8}")
    print("-" * 55)
    auc, fpr, f1, cov = evaluate_text(project_root, args.text_ckpt, use_knn_star=False)
    print(f"{1:<3} {'Text-BERT':<18} {fpr:8.2f} {auc:8.2f} {f1:8.2f} {cov:8.2f}")
    auc, fpr, f1, cov = evaluate_multi(project_root, args.multi_ckpt, use_knn_star=False)
    print(f"{2:<3} {'LayoutLMv3':<18} {fpr:8.2f} {auc:8.2f} {f1:8.2f} {cov:8.2f}")

    print("\nPhase 3 — OOD comparison (KNN*)")
    print(f"{'#':<3} {'Model':<18} {'FPR↓':>8} {'AUC↑':>8} {'F1↑':>8} {'COV↑':>8}")
    print("-" * 55)
    auc, fpr, f1, cov = evaluate_text(project_root, args.text_ckpt, use_knn_star=True)
    print(f"{1:<3} {'Text-BERT':<18} {fpr:8.2f} {auc:8.2f} {f1:8.2f} {cov:8.2f}")
    auc, fpr, f1, cov = evaluate_multi(project_root, args.multi_ckpt, use_knn_star=True)
    print(f"{2:<3} {'LayoutLMv3':<18} {fpr:8.2f} {auc:8.2f} {f1:8.2f} {cov:8.2f}")


if __name__ == "__main__":
    main()