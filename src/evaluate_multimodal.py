import argparse
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

from config import Paths, TrainConfig, OODConfig
from data import set_seed
from data_multimodal import RVLCDIPLayoutLMv3Dataset
from model_multimodal import LayoutLMv3DocClassifier
from knn_ood_multimodal import (
    extract_embeddings_and_logits_multimodal,
    build_faiss_l2_index,
    estimate_threshold_theta,
    knn_predict_no_agreement,
    knn_star_predict,
    knn1_score_and_neighbor,
)
from metrics import compute_auc, compute_fpr_at_tpr95, compute_end_to_end_metrics


def compute_scores(train_index, embeddings: np.ndarray) -> np.ndarray:
    scores = np.zeros((embeddings.shape[0],), dtype=np.float32)
    for i in range(embeddings.shape[0]):
        s, _, _ = knn1_score_and_neighbor(train_index, embeddings[i])
        scores[i] = s
    return scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project_root", type=str, required=True)
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--use_knn_star", action="store_true")
    args = ap.parse_args()

    cfg = TrainConfig()
    ood_cfg = OODConfig()
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

    max_length = ckpt["max_length"]

    # Use debug subset if debug checkpoint, else full
    is_debug = Path(args.ckpt).stem.endswith("_debug")
    debug_samples = cfg.debug_samples if is_debug else None

    train_ds = RVLCDIPLayoutLMv3Dataset(
        rvl_root=paths.rvl_cdip_dir,
        ocr_box_root=ocr_box_root,
        split_file=paths.train_list,
        processor_name=ckpt["model_name"],
        max_length=max_length,
        debug_samples=debug_samples,
    )
    val_ds = RVLCDIPLayoutLMv3Dataset(
        rvl_root=paths.rvl_cdip_dir,
        ocr_box_root=ocr_box_root,
        split_file=paths.val_list,
        processor_name=ckpt["model_name"],
        max_length=max_length,
        debug_samples=debug_samples,
    )
    test_ds = RVLCDIPLayoutLMv3Dataset(
        rvl_root=paths.rvl_cdip_dir,
        ocr_box_root=ocr_box_root,
        split_file=paths.test_list,
        processor_name=ckpt["model_name"],
        max_length=max_length,
        debug_samples=debug_samples,
    )

    # For OOD, reuse your image paths but boxes come from rvl-cdip-o-text replacement later if needed.
    # Phase 2 assumes you also generated QS-OCR-BOX-OOD if you want full multimodal OOD.
    # For now, this script uses only ID closed/open-set if OOD boxes exist.
    ood_box_root = project_root / "QS-OCR-BOX-OOD"
    from data_multimodal_ood import RVLCDIPOODLayoutLMv3Dataset

    ood_ds = RVLCDIPOODLayoutLMv3Dataset(
        rvl_ood_root=project_root / "rvl-cdip-o",
        ocr_box_root=ood_box_root,
        split_dir=paths.rvl_cdip_ood_text_dir,
        processor_name=ckpt["model_name"],
        max_length=max_length,
        debug_samples=debug_samples,
    )

    train_loader = DataLoader(train_ds, batch_size=8, shuffle=False, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=8, shuffle=False, num_workers=0)
    ood_loader = DataLoader(ood_ds, batch_size=8, shuffle=False, num_workers=0)

    train_emb, train_logits, train_y = extract_embeddings_and_logits_multimodal(model, train_loader, device=str(device))
    val_emb, val_logits, val_y = extract_embeddings_and_logits_multimodal(model, val_loader, device=str(device))
    test_emb, test_logits, test_y = extract_embeddings_and_logits_multimodal(model, test_loader, device=str(device))
    ood_emb, ood_logits, ood_y = extract_embeddings_and_logits_multimodal(model, ood_loader, device=str(device))

    train_index = build_faiss_l2_index(train_emb.astype(np.float32), train_y.astype(np.int64))

    theta = estimate_threshold_theta(train_index, val_emb.astype(np.float32), tpr_target=ood_cfg.tpr_target)
    print(f"Theta (TPR>={ood_cfg.tpr_target} on VAL): {theta:.6f}")

    id_scores = compute_scores(train_index, test_emb.astype(np.float32))
    ood_scores = compute_scores(train_index, ood_emb.astype(np.float32))

    auc = compute_auc(id_scores, ood_scores)
    fpr95 = compute_fpr_at_tpr95(id_scores, ood_scores, tpr_target=ood_cfg.tpr_target)
    print(f"AUC: {auc:.4f}")
    print(f"FPR@TPR95: {fpr95:.4f}")

    if args.use_knn_star:
        is_id_test, pred_test = knn_star_predict(train_index, test_emb, test_logits, theta)
        is_id_ood, pred_ood = knn_star_predict(train_index, ood_emb, ood_logits, theta)
        print("Using KNN* (agreement).")
    else:
        is_id_test, pred_test = knn_predict_no_agreement(train_index, test_emb, test_logits, theta)
        is_id_ood, pred_ood = knn_predict_no_agreement(train_index, ood_emb, ood_logits, theta)
        print("Using KNN.")

    e2e = compute_end_to_end_metrics(
        y_true_id=test_y,
        is_id_pred_idset=is_id_test,
        y_pred_idset=pred_test,
        is_id_pred_oodset=is_id_ood,
    )

    print("\nMultimodal Phase 2 end-to-end metrics")
    print(f"PRE: {e2e.pre:.4f}")
    print(f"REC: {e2e.rec:.4f}")
    print(f"F1 : {e2e.f1:.4f}")
    print(f"COV: {e2e.cov:.4f}")
    print(f"Counts: TP={e2e.tp}, FN={e2e.fn}, FP={e2e.fp}, TN={e2e.tn}, TP_correct={e2e.tp_correct}")


if __name__ == "__main__":
    main()