import argparse
from pathlib import Path
import subprocess
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project_root", required=True)
    ap.add_argument("--text_ckpt", required=True)
    ap.add_argument("--multi_ckpt", required=True)
    args = ap.parse_args()

    print("\nPhase 3 — Precision-threshold comparison")
    print("Run these two commands separately for exact PRE-controlled comparison:\n")

    print("Text-only BERT:")
    print(
        f'python src/novelty_detecton_result_99PRE.py --project_root "{args.project_root}" '
        f'--ckpt_dir "{Path(args.text_ckpt).parent}" --suffix {"debug" if Path(args.text_ckpt).stem.endswith("_debug") else "full"}'
    )

    print("\nMultimodal LayoutLMv3:")
    print(
        f'python src/novelty_detection_result_99PRE_multimodal.py --project_root "{args.project_root}" '
        f'--ckpt "{args.multi_ckpt}"'
    )


if __name__ == "__main__":
    main()