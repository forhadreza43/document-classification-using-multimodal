from PIL import Image
import pytesseract
import os

# Point pytesseract to the Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

input_folder = "rvl-cdip-o"
output_folder = "rvl-cdip-o-text"

os.makedirs(output_folder, exist_ok=True)


def extract_text_from_image():
    for file in os.listdir(input_folder):
        if file.endswith(".tif"):
            img = Image.open(os.path.join(input_folder, file))
            text = pytesseract.image_to_string(img)

            txt_name = file.replace(".tif", ".txt")
            with open(os.path.join(output_folder, txt_name), "w", encoding="utf-8") as f:
                f.write(text)


def main():
    extract_text_from_image()


if __name__ == "__main__":
    main()
