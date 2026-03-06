import argparse
from pathlib import Path
import pytesseract
from PIL import Image
from tqdm import tqdm


def read_split(split_file):
    items = []
    with open(split_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 1:
                items.append(parts[0])
    return items


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--project_root", type=str, required=True)
    parser.add_argument("--tesseract_cmd", type=str, default=None)
    args = parser.parse_args()

    root = Path(args.project_root)

    rvl_dir = root / "rvl-cdip"
    txt_dir = root / "QS-OCR-Large"

    splits = [
        root / "train.txt",
        root / "val.txt",
        root / "test.txt"
    ]

    if args.tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = args.tesseract_cmd

    files = []

    for split in splits:
        files.extend(read_split(split))

    files = list(set(files))

    print("Total documents referenced:", len(files))

    missing = []

    for rel_path in files:

        txt_path = txt_dir / rel_path.replace(".tif", ".txt")

        if not txt_path.exists():
            missing.append(rel_path)

    print("Missing OCR files:", len(missing))

    for rel_path in tqdm(missing):

        img_path = rvl_dir / rel_path
        txt_path = txt_dir / rel_path.replace(".tif", ".txt")

        if not img_path.exists():
            continue

        txt_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            image = Image.open(img_path)
            text = pytesseract.image_to_string(image)

            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(text)

        except Exception as e:
            print("OCR error:", img_path, e)


if __name__ == "__main__":
    main()