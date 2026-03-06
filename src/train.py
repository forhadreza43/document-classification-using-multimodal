import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from collections import Counter

from config import Paths, TrainConfig, LossConfig
from data import RVLCDIPOCRTextDataset, set_seed
from sampler import MinPerClassBatchSampler
from model import BertDocClassifier

# Keep your Margin* loss exactly
from loss import CustomMarginContrastiveLoss

# Extra losses (CE, fixed-margin, SCL, weighted CE)
from losses_extra import CELoss, FixedMarginLoss, SCLLoss, WeightedCELoss


def build_criterion(loss_name: str, loss_cfg: LossConfig, train_ds: RVLCDIPOCRTextDataset, num_classes: int, device):
    """
    Returns a callable criterion with signature (logits, h, labels) -> loss.
    """
    if loss_name == "margin_star":
        # Your original loss (Margin*)
        return CustomMarginContrastiveLoss(
            alpha=loss_cfg.alpha, beta=loss_cfg.beta, lam=loss_cfg.lam, eps=loss_cfg.eps
        )

    if loss_name == "margin":
        # Fixed margin (Margin)
        return FixedMarginLoss(
            alpha=loss_cfg.alpha, beta=loss_cfg.beta, lam=loss_cfg.lam, eps=loss_cfg.eps
        )

    if loss_name == "scl":
        # CE + supervised contrastive
        return SCLLoss(temperature=0.1, lam=1.0)

    if loss_name == "weight":
        # Weighted CE from training set distribution
        counts = Counter([y for _, y in train_ds.items])
        weights = np.zeros((num_classes,), dtype=np.float32)
        for c in range(num_classes):
            # inverse frequency; keep stable if missing in debug subset
            weights[c] = 1.0 / max(1, counts.get(c, 1))
        weights = weights / weights.mean()
        class_w = torch.tensor(weights, dtype=torch.float32, device=device)
        return WeightedCELoss(class_weights=class_w)

    if loss_name == "ce":
        return CELoss()

    raise ValueError(f"Unknown loss: {loss_name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project_root", type=str, required=True)
    ap.add_argument("--save_dir", type=str, default="checkpoints")
    ap.add_argument("--loss", type=str, default="margin_star",
                    choices=["margin_star", "margin", "scl", "weight", "ce"])
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
    loss_cfg = LossConfig()

    set_seed(cfg.seed)
    device = torch.device(cfg.device)

    # Only restrict labels in debug mode to keep sampler feasible.
    allowed = None
    if cfg.debug_samples is not None:
        tmp_ds = RVLCDIPOCRTextDataset(
            qs_root=paths.qs_ocr_large_dir,
            split_file=paths.train_list,
            tokenizer_name=cfg.model_name,
            max_length=cfg.max_length,
            debug_samples=cfg.debug_samples,
        )
        labels_all = [int(tmp_ds.items[i][1]) for i in range(len(tmp_ds))]
        cnt = Counter(labels_all)
        allowed = {lab for lab, c in cnt.items() if c >= cfg.min_per_class}
        if len(allowed) == 0:
            raise RuntimeError(
                "Debug subset has no label with >= min_per_class samples. "
                "Increase debug_samples or reduce min_per_class for the debug run."
            )

    train_ds = RVLCDIPOCRTextDataset(
        qs_root=paths.qs_ocr_large_dir,
        split_file=paths.train_list,
        tokenizer_name=cfg.model_name,
        max_length=cfg.max_length,
        debug_samples=cfg.debug_samples,
        allowed_labels=allowed,
    )

    labels = [int(y) for (_, y) in train_ds.items]

    # RVL-CDIP is always 16 classes (0..15)
    num_classes = 16

    batch_sampler = MinPerClassBatchSampler(labels, cfg.batch_size, cfg.min_per_class, seed=cfg.seed)
    train_loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=cfg.num_workers)

    print("Loss:", args.loss)
    print("Train dataset size:", len(train_ds))
    print("Num unique labels:", len(set([y for _, y in train_ds.items])))
    print("Min label:", min([y for _, y in train_ds.items]), "Max label:", max([y for _, y in train_ds.items]))
    print("Train loader len (num batches):", len(train_loader))

    model = BertDocClassifier(cfg.model_name, num_classes=num_classes).to(device)
    criterion = build_criterion(args.loss, loss_cfg, train_ds, num_classes, device=device)

    optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    model.train()
    for epoch in range(cfg.epochs):
        pbar = tqdm(train_loader, desc=f"Train epoch {epoch + 1}/{cfg.epochs}")
        for step, batch in enumerate(pbar):
            if step == 0:
                print(">>> entered training loop, got first batch")

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch.get("token_type_ids")
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device)
            labels_t = batch["labels"].to(device)

            # Safety assert for RVL-CDIP
            if labels_t.min().item() < 0 or labels_t.max().item() > 15:
                raise ValueError(
                    f"Found label outside 0..15: min={labels_t.min().item()} max={labels_t.max().item()}"
                )

            optim.zero_grad(set_to_none=True)
            logits, h = model(input_ids, attention_mask, token_type_ids)
            loss = criterion(logits, h, labels_t)
            loss.backward()
            optim.step()

            pbar.set_postfix({"loss": float(loss.detach().cpu())})

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    suffix = "debug" if cfg.debug_samples is not None else "full"
    ckpt_path = save_dir / f"bert_{args.loss}_{suffix}.pt"

    torch.save(
        {
            "model_state": model.state_dict(),
            "num_classes": num_classes,
            "model_name": cfg.model_name,
            "max_length": cfg.max_length,
        },
        ckpt_path,
    )
    print(f"Saved checkpoint to {ckpt_path}")


if __name__ == "__main__":
    main()