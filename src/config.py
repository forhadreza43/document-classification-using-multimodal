from dataclasses import dataclass
from pathlib import Path

@dataclass
class Paths:
    project_root: Path
    qs_ocr_large_dir: Path        # Project/QS-OCR-Large
    rvl_cdip_dir: Path            # Project/rvl-cdip (not strictly used in text-only)
    rvl_cdip_ood_text_dir: Path   # Project/rvl-cdip-o-text
    train_list: Path              # Project/train.txt
    val_list: Path                # Project/val.txt
    test_list: Path               # Project/test.txt

@dataclass
class TrainConfig:
    model_name: str = "bert-base-uncased"  # paper uses BERT for English
    max_length: int = 128                  # GPU run: increase to 512, paper uses 512
    batch_size: int = 8                    # GPU run: increase to 128
    min_per_class: int = 4                 # GPU run: increase to 8, paper: ensure enough positives per class in batch
    lr: float = 2e-5
    weight_decay: float = 0.01
    epochs: int = 1                        # CPU debug: keep 1; GPU run: increase (3-5)
    seed: int = 42
    num_workers: int = 0                   # Windows/PyCharm friendly
    debug_samples: int = 128               # GPU run: None
    device: str = "cpu"                    # GPU run: switch to "cuda" later

@dataclass
class LossConfig:
    alpha: float = 1.5   # Margin*
    beta: float = 0.5
    lam: float = 4.0
    eps: float = 1e-12

@dataclass
class OODConfig:
    k: int = 1
    tpr_target: float = 0.95