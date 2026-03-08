import argparse
from pathlib import Path
from collections import Counter

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import TrainConfig, Paths
from sampler import MinPerClassBatchSampler
from data import set_seed
from data_multimodal import RVLCDIPLayoutLMv3Dataset
from model_multimodal import LayoutLMv3DocClassifier


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project_root", type=str, required=True)
    ap.add_argument("--save_dir", type=str, default="checkpoints")
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

    allowed = None
    if cfg.debug_samples is not None:
        tmp_ds = RVLCDIPLayoutLMv3Dataset(
            rvl_root=paths.rvl_cdip_dir,
            ocr_box_root=ocr_box_root,
            split_file=paths.train_list,
            processor_name="microsoft/layoutlmv3-base",
            max_length=cfg.max_length,
            debug_samples=cfg.debug_samples,
        )
        labels_all = [tmp_ds.items[i][2] for i in range(len(tmp_ds))]
        cnt = Counter(labels_all)
        allowed = {lab for lab, c in cnt.items() if c >= cfg.min_per_class}
        if len(allowed) == 0:
            raise RuntimeError("No label has enough samples in debug subset.")

    train_ds = RVLCDIPLayoutLMv3Dataset(
        rvl_root=paths.rvl_cdip_dir,
        ocr_box_root=ocr_box_root,
        split_file=paths.train_list,
        processor_name="microsoft/layoutlmv3-base",
        max_length=cfg.max_length,
        debug_samples=cfg.debug_samples,
        allowed_labels=allowed,
    )

    val_ds = RVLCDIPLayoutLMv3Dataset(
        rvl_root=paths.rvl_cdip_dir,
        ocr_box_root=ocr_box_root,
        split_file=paths.val_list,
        processor_name="microsoft/layoutlmv3-base",
        max_length=cfg.max_length,
        debug_samples=cfg.debug_samples,
        allowed_labels=allowed,
    )

    labels = [y for _, _, y in train_ds.items]
    batch_sampler = MinPerClassBatchSampler(labels, cfg.batch_size, cfg.min_per_class, seed=cfg.seed)

    train_loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=cfg.num_workers)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)

    model = LayoutLMv3DocClassifier("microsoft/layoutlmv3-base", num_classes=16).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    best_acc = -1.0
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = save_dir / ("layoutlmv3_debug.pt" if cfg.debug_samples is not None else "layoutlmv3_full.pt")

    for epoch in range(cfg.epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"Train epoch {epoch+1}/{cfg.epochs}")

        for batch in pbar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            bbox = batch["bbox"].to(device)
            pixel_values = batch["pixel_values"].to(device)
            labels_t = batch["labels"].to(device)

            optim.zero_grad(set_to_none=True)
            logits, h = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                bbox=bbox,
                pixel_values=pixel_values,
            )
            loss = criterion(logits, labels_t)
            loss.backward()
            optim.step()
            pbar.set_postfix({"loss": float(loss.detach().cpu())})

        # validation ACC
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                bbox = batch["bbox"].to(device)
                pixel_values = batch["pixel_values"].to(device)
                labels_t = batch["labels"].to(device)

                logits, _ = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    bbox=bbox,
                    pixel_values=pixel_values,
                )
                pred = logits.argmax(dim=1)
                correct += int((pred == labels_t).sum().item())
                total += int(labels_t.size(0))

        acc = correct / max(1, total)
        print(f"Validation ACC: {acc*100:.2f}")

        if acc > best_acc:
            best_acc = acc
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "model_name": "microsoft/layoutlmv3-base",
                    "num_classes": 16,
                    "max_length": cfg.max_length,
                },
                ckpt_path,
            )
            print(f"Saved best checkpoint to {ckpt_path}")


if __name__ == "__main__":
    main()