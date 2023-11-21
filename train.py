import os

import numpy as np
import progressbar
import tensorflow as tf
from PIL import Image
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import get_custom_objects

import settings
from model import barcode_model

gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)


def get_numpy_img_data(path: str):
    img = Image.open(path).convert("L")

    return np.asarray(img)


def read_data():
    input_d = []
    output_d = []

    for output_filename in progressbar.progressbar(
        os.listdir(settings.BARCODE_OUTPUT_PATH)
    ):
        for input_filename in os.listdir(
            f"{settings.BARCODE_INPUT_PATH}/{output_filename.split('.')[0]}"
        ):
            output_d.append(
                get_numpy_img_data(
                    path=f"{settings.BARCODE_OUTPUT_PATH}/{output_filename}"
                )
            )
            input_d.append(
                get_numpy_img_data(
                    path=f"{settings.BARCODE_INPUT_PATH}/{output_filename.split('.')[0]}/{input_filename}"
                )
            )

    return input_d, output_d


def sharpening_loss(y_true, y_pred):
    true_gradients = tf.image.sobel_edges(y_true)
    pred_gradients = tf.image.sobel_edges(y_pred)

    edge_loss = tf.reduce_mean(tf.square(true_gradients - pred_gradients))
    intensity_loss = tf.reduce_mean(tf.abs(y_true - y_pred))

    return 0.3 * edge_loss + 0.7 * intensity_loss


input_data, output_data = read_data()

input_data = np.expand_dims(input_data, axis=-1)
output_data = np.expand_dims(output_data, axis=-1)

input_data = input_data / 255.0
output_data = output_data / 255.0

train_data = input_data[:800]
train_labels = output_data[:800]
val_data = input_data[:200]
val_labels = output_data[:200]


get_custom_objects().update({"sharpening_loss": sharpening_loss})


if os.path.exists("best_model.h5"):
    nn_model = load_model(
        "best_model.h5", custom_objects={"sharpening_loss": sharpening_loss}
    )
else:
    nn_model = barcode_model()


nn_model.compile(optimizer=Adam(lr=1e-4), loss=sharpening_loss)

checkpoint = ModelCheckpoint("best_model.h5", verbose=1, save_best_only=True)
early_stopping = EarlyStopping(patience=10, verbose=1)

nn_model.fit(
    train_data,
    train_labels,
    epochs=50,
    batch_size=2,
    validation_data=(val_data, val_labels),
    callbacks=[checkpoint, early_stopping],
)
