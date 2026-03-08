import argparse
from pathlib import Path
from typing import Optional, List, Tuple

import pytesseract
from PIL import Image, ImageSequence
from tqdm import tqdm


def iter_tiff_pages(img_path: Path, max_pages: Optional[int] = None):
    with Image.open(img_path) as img:
        for i, frame in enumerate(ImageSequence.Iterator(img)):
            if max_pages is not None and i >= max_pages:
                break
            yield frame.convert("RGB")


def safe_int(x, default=0):
    try:
        return int(float(x))
    except Exception:
        return default


def extract_word_boxes_from_page(page_image: Image.Image, lang: str, psm: int, oem: int):
    """
    Returns list of tuples:
      (word, x1, y1, x2, y2)
    Coordinates are in original page pixel space.
    """
    config = f"--oem {oem} --psm {psm}"
    data = pytesseract.image_to_data(
        page_image,
        lang=lang,
        config=config,
        output_type=pytesseract.Output.DICT,
    )

    n = len(data["text"])
    rows = []

    for i in range(n):
        text = data["text"][i].strip()
        conf = safe_int(data["conf"][i], default=-1)

        if not text:
            continue
        if conf < 0:
            continue

        x = safe_int(data["left"][i])
        y = safe_int(data["top"][i])
        w = safe_int(data["width"][i])
        h = safe_int(data["height"][i])

        if w <= 0 or h <= 0:
            continue

        x1, y1 = x, y
        x2, y2 = x + w, y + h
        rows.append((text, x1, y1, x2, y2))

    return rows


def ensure_parent(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)


def write_tsv(tsv_path: Path, rows: List[Tuple[str, int, int, int, int]]):
    ensure_parent(tsv_path)
    with tsv_path.open("w", encoding="utf-8", errors="ignore") as f:
        f.write("word\tx1\ty1\tx2\ty2\n")
        for word, x1, y1, x2, y2 in rows:
            word = word.replace("\t", " ").replace("\n", " ").replace("\r", " ").strip()
            if not word:
                continue
            f.write(f"{word}\t{x1}\t{y1}\t{x2}\t{y2}\n")


def collect_ood_images(ood_root: Path) -> List[Path]:
    tif_files = sorted(ood_root.glob("*.tif"))
    # also accept .tiff just in case
    tif_files += sorted(ood_root.glob("*.tiff"))
    # de-duplicate while preserving order
    seen = set()
    unique = []
    for p in tif_files:
        key = p.resolve()
        if key not in seen:
            seen.add(key)
            unique.append(p)
    return unique


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project_root", type=str, required=True, help=r'Example: "D:\Thesis\Project"')
    ap.add_argument("--ood_dir", type=str, default="rvl-cdip-o", help="Flat folder containing OOD *.tif files")
    ap.add_argument("--out_dir", type=str, default="QS-OCR-BOX-OOD", help="Output folder for OOD TSV boxes")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing TSV files")
    ap.add_argument("--missing_only", action="store_true", help="Process only missing TSVs")
    ap.add_argument("--limit", type=int, default=None, help="Only process first N images")
    ap.add_argument("--lang", type=str, default="eng")
    ap.add_argument("--psm", type=int, default=6, help="Tesseract page segmentation mode")
    ap.add_argument("--oem", type=int, default=3, help="Tesseract OCR engine mode")
    ap.add_argument("--max_pages", type=int, default=1, help="Max TIFF pages to OCR; default 1")
    ap.add_argument("--tesseract_cmd", type=str, default=None, help=r'Example: "C:\Program Files\Tesseract-OCR\tesseract.exe"')
    args = ap.parse_args()

    project_root = Path(args.project_root)
    ood_root = project_root / args.ood_dir
    out_root = project_root / args.out_dir

    if args.tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = args.tesseract_cmd

    if not ood_root.exists():
        raise FileNotFoundError(f"OOD folder not found: {ood_root}")

    img_files = collect_ood_images(ood_root)
    if args.limit is not None:
        img_files = img_files[:args.limit]

    print(f"OOD images found: {len(img_files)}")
    print(f"OOD image root: {ood_root}")
    print(f"Output TSV root: {out_root}")

    to_process = []
    existing_tsv = 0

    for img_path in img_files:
        tsv_path = out_root / f"{img_path.stem}.tsv"

        if tsv_path.exists():
            existing_tsv += 1
            if args.missing_only and not args.overwrite:
                continue
            if (not args.overwrite) and (not args.missing_only):
                continue

        to_process.append((img_path, tsv_path))

    print(f"Existing TSVs: {existing_tsv}")
    print(f"Will process: {len(to_process)}")

    processed = 0
    errors = 0
    empty_docs = 0

    for img_path, tsv_path in tqdm(to_process, desc="Generating OOD OCR boxes"):
        try:
            all_rows = []

            for page in iter_tiff_pages(img_path, max_pages=args.max_pages):
                rows = extract_word_boxes_from_page(
                    page_image=page,
                    lang=args.lang,
                    psm=args.psm,
                    oem=args.oem,
                )
                all_rows.extend(rows)

            if len(all_rows) == 0:
                empty_docs += 1

            write_tsv(tsv_path, all_rows)
            processed += 1

        except Exception as e:
            errors += 1
            ensure_parent(tsv_path)
            with tsv_path.open("w", encoding="utf-8", errors="ignore") as f:
                f.write("word\tx1\ty1\tx2\ty2\n")
            print(f"\n[ERROR] {img_path}: {repr(e)}")

    print("\nDone.")
    print(f"Processed: {processed}")
    print(f"Errors: {errors}")
    print(f"Empty docs: {empty_docs}")
    print(f"Output root: {out_root}")


if __name__ == "__main__":
    main()