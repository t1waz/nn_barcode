import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import get_custom_objects
import settings


def sharpening_loss(y_true, y_pred):
    # edge gradient
    true_gradients = tf.image.sobel_edges(y_true)
    pred_gradients = tf.image.sobel_edges(y_pred)

    # edge sharpen
    edge_loss = tf.reduce_mean(tf.square(true_gradients - pred_gradients))

    intensity_loss = tf.reduce_mean(tf.abs(y_true - y_pred))
    loss = edge_loss + intensity_loss

    return loss


get_custom_objects().update({"sharpening_loss": sharpening_loss})
model = tf.keras.models.load_model("best_model.h5")


def load_image(image_file):
    img = Image.open(image_file).convert("L")
    img = img.resize(settings.BARCODE_SIZE)
    return np.array(img)


def predict_and_enhance(image):
    img_array = np.expand_dims(image, axis=0)
    img_array = np.expand_dims(img_array, axis=-1)
    img_array = img_array / 255.0

    prediction = model.predict(img_array)
    return prediction[0, :, :, 0]


st.title("NN Barcode")

uploaded_file = st.file_uploader("Pickup image", type=["png", "jpg", "jpeg"])


if uploaded_file is not None:
    image = load_image(uploaded_file)
    st.image(image, caption="source img", use_column_width=True)

    if st.button("sharpen"):
        enhanced_image = predict_and_enhance(image)
        st.image(enhanced_image, caption="target img", use_column_width=True)
