import argparse
from pathlib import Path
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
    knn1_score_and_neighbor
)
from metrics import compute_auc, compute_fpr_at_tpr95, compute_end_to_end_metrics

def compute_scores_from_index(train_index, embeddings: np.ndarray) -> np.ndarray:
    scores = np.zeros((embeddings.shape[0],), dtype=np.float32)
    for i in range(embeddings.shape[0]):
        s, _, _ = knn1_score_and_neighbor(train_index, embeddings[i])
        scores[i] = s
    return scores

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project_root", type=str, required=True)
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--use_knn_star", action="store_true", help="Use KNN* (agreement). Default is KNN (no agreement).")
    args = ap.parse_args()

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

    cfg = TrainConfig()
    ood_cfg = OODConfig()
    set_seed(cfg.seed)
    device = torch.device(cfg.device)

    ckpt = torch.load(args.ckpt, map_location="cpu")
    model_name = ckpt["model_name"]
    # num_classes = ckpt["num_classes"]
    # label_to_new = ckpt["label_to_new"]
    max_length = ckpt["max_length"]
    num_classes = 16

    model = BertDocClassifier(model_name, num_classes=num_classes).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # --- Datasets: in-distribution val/test and OOD ---
    # For quick CPU test: use debug_samples for all sets.
    val_ds = RVLCDIPOCRTextDataset(
        qs_root=paths.qs_ocr_large_dir,
        split_file=paths.val_list,
        tokenizer_name=model_name,
        max_length=max_length,
        debug_samples=cfg.debug_samples,
    )
    test_ds = RVLCDIPOCRTextDataset(
        qs_root=paths.qs_ocr_large_dir,
        split_file=paths.test_list,
        tokenizer_name=model_name,
        max_length=max_length,
        debug_samples=cfg.debug_samples,
    )
    ood_ds = RVLCDIPOODTextDataset(
        ood_text_dir=paths.rvl_cdip_ood_text_dir,
        tokenizer_name=model_name,
        max_length=max_length,
        debug_samples=cfg.debug_samples,
    )

    # Apply same label remap as training (unknown labels in debug subset are dropped for fairness)
    # def remap_inplace(ds):
    #     new_items = []
    #     for p, y in ds.items:
    #         if y in label_to_new:
    #             new_items.append((p, label_to_new[y]))
    #     ds.items = new_items
    #
    # remap_inplace(val_ds)
    # remap_inplace(test_ds)

    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False, num_workers=0)
    ood_loader = DataLoader(ood_ds, batch_size=16, shuffle=False, num_workers=0)

    # --- Build FAISS index from TRAIN embeddings ---
    # For evaluation sanity test we index the TRAIN DEBUG subset too (consistent with your constraint).
    # train_ds = RVLCDIPOCRTextDataset(
    #     qs_root=paths.qs_ocr_large_dir,
    #     split_file=paths.train_list,
    #     tokenizer_name=model_name,
    #     max_length=max_length,
    #     debug_samples=cfg.debug_samples,
    #     allowed_labels=set(label_to_new.keys()),
    # )
    # train_ds.items = [(p, label_to_new[y]) for (p, y) in train_ds.items]
    # train_loader = DataLoader(train_ds, batch_size=16, shuffle=False, num_workers=0)

    train_ds = RVLCDIPOCRTextDataset(
        qs_root=paths.qs_ocr_large_dir,
        split_file=paths.train_list,
        tokenizer_name=model_name,
        max_length=max_length,
        debug_samples=cfg.debug_samples,
    )
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=False, num_workers=0)

    train_emb, train_logits, train_y = extract_embeddings_and_logits(model, train_loader, device=str(device))
    train_index = build_faiss_l2_index(train_emb.astype(np.float32), train_y.astype(np.int64))

    # --- Extract val/test/OOD embeddings+logits ---
    val_emb, val_logits, val_y = extract_embeddings_and_logits(model, val_loader, device=str(device))
    test_emb, test_logits, test_y = extract_embeddings_and_logits(model, test_loader, device=str(device))
    ood_emb, ood_logits, ood_y = extract_embeddings_and_logits(model, ood_loader, device=str(device))

    # --- Threshold theta from VAL (paper: use ID set to satisfy TPR target) ---
    theta = estimate_threshold_theta(train_index, val_emb.astype(np.float32), tpr_target=ood_cfg.tpr_target)
    print(f"Theta (TPR>={ood_cfg.tpr_target} on VAL): {theta:.6f}")

    # --- OOD scores for AUC/FPR@TPR95 ---
    id_scores = compute_scores_from_index(train_index, test_emb.astype(np.float32))
    ood_scores = compute_scores_from_index(train_index, ood_emb.astype(np.float32))

    auc = compute_auc(id_scores, ood_scores)
    fpr95 = compute_fpr_at_tpr95(id_scores, ood_scores, tpr_target=ood_cfg.tpr_target)
    print(f"AUC: {auc:.4f}")
    print(f"FPR@TPR{int(ood_cfg.tpr_target*100)}: {fpr95:.4f}")

    # --- Full end-to-end pipeline ---
    if args.use_knn_star:
        is_id_test, pred_test = knn_star_predict(train_index, test_emb, test_logits, theta)
        is_id_ood, pred_ood = knn_star_predict(train_index, ood_emb, ood_logits, theta)
        print("Using KNN* (agreement) for ambiguity rejection.")
    else:
        is_id_test, pred_test = knn_predict_no_agreement(train_index, test_emb, test_logits, theta)
        is_id_ood, pred_ood = knn_predict_no_agreement(train_index, ood_emb, ood_logits, theta)
        print("Using KNN (no agreement).")

    e2e = compute_end_to_end_metrics(
        y_true_id=test_y,
        is_id_pred_idset=is_id_test,
        y_pred_idset=pred_test,
        is_id_pred_oodset=is_id_ood
    )

    print("\nEnd-to-end metrics:")
    print(f"  PRE: {e2e.pre:.4f}")
    print(f"  REC: {e2e.rec:.4f}")
    print(f"  F1 : {e2e.f1:.4f}")
    print(f"  COV: {e2e.cov:.4f}")
    print(f"  Counts: TP={e2e.tp}, FN={e2e.fn}, FP={e2e.fp}, TN={e2e.tn}, TP_correct={e2e.tp_correct}")

if __name__ == "__main__":
    main()