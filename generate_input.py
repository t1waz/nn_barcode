import os

import progressbar
from PIL import Image, ImageFilter

import settings
from utils import create_dir


def generate_input(filename) -> None:
    input_folder = filename.split(".")[0]
    create_dir(dir_path=f"{settings.BARCODE_INPUT_PATH}/{input_folder}")

    output_img = Image.open(f"{settings.BARCODE_OUTPUT_PATH}/{filename}")

    blur_normal = output_img.filter(ImageFilter.BLUR)
    blur_box = output_img.filter(ImageFilter.BoxBlur(8))
    blur_gaussian = output_img.filter(ImageFilter.GaussianBlur(3))

    blur_box = blur_box.resize(settings.BARCODE_SIZE, Image.Resampling.BILINEAR)
    blur_normal = blur_normal.resize(settings.BARCODE_SIZE, Image.Resampling.BILINEAR)
    blur_gaussian = blur_gaussian.resize(
        settings.BARCODE_SIZE, Image.Resampling.BILINEAR
    )

    blur_box.save(f"{settings.BARCODE_INPUT_PATH}/{input_folder}/box.png")
    blur_normal.save(f"{settings.BARCODE_INPUT_PATH}/{input_folder}/normal.png")
    blur_gaussian.save(f"{settings.BARCODE_INPUT_PATH}/{input_folder}/gaussian.png")


for filename in progressbar.progressbar(os.listdir(settings.BARCODE_OUTPUT_PATH)):
    generate_input(filename=filename)
