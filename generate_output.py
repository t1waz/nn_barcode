import random
import string

import barcode
import progressbar
from barcode.writer import ImageWriter

import settings
from PIL import Image
import io
from utils import create_dir


SEPARATOR_CHARS = ["-", ":", ";", "|"]

BARCODE_128 = barcode.get_barcode_class("code128")


def get_random_string(length: int) -> str:
    return "".join(random.choice(string.ascii_uppercase) for _ in range(length))


def generate_fake_sign(seg_min_len: int, seg_max_len: int, segments: int):
    return random.choice(SEPARATOR_CHARS).join(
        get_random_string(length=random.randint(seg_min_len, seg_max_len))
        for _ in range(segments)
    )


def generate_barcode(msg: str) -> None:
    new_barcode = BARCODE_128(msg, writer=ImageWriter())
    barcode_bytes = io.BytesIO()
    new_barcode.write(barcode_bytes)
    pil_image = Image.open(barcode_bytes)
    pil_image = pil_image.resize(settings.BARCODE_SIZE, Image.Resampling.BILINEAR)
    create_dir(dir_path=f"{settings.BARCODE_OUTPUT_PATH}")
    pil_image.save(f"{settings.BARCODE_OUTPUT_PATH}/{msg}.png")


for _ in progressbar.progressbar(range(1000)):
    msg = generate_fake_sign(1, 5, random.randint(3, 5))
    generate_barcode(msg=msg)
