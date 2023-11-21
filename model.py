from tensorflow.keras.layers import (
    Input,
    Conv2D,
    MaxPooling2D,
    UpSampling2D,
    Concatenate,
    BatchNormalization,
    LeakyReLU,
    Dropout,
)
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import HeNormal


def barcode_model(input_size=(512, 512, 1), dropout_rate=0.3):
    inputs = Input(input_size)

    # Encoder
    conv1 = Conv2D(
        32, (3, 3), activation="relu", padding="same", kernel_initializer=HeNormal()
    )(inputs)
    conv1 = BatchNormalization()(conv1)
    conv1 = LeakyReLU(alpha=0.1)(conv1)
    conv1 = Conv2D(
        32, (3, 3), activation="relu", padding="same", kernel_initializer=HeNormal()
    )(conv1)
    conv1 = BatchNormalization()(conv1)
    conv1 = LeakyReLU(alpha=0.1)(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    pool1 = Dropout(dropout_rate)(pool1)

    # Bottleneck
    conv2 = Conv2D(
        64, (3, 3), activation="relu", padding="same", kernel_initializer=HeNormal()
    )(pool1)
    conv2 = BatchNormalization()(conv2)
    conv2 = LeakyReLU(alpha=0.1)(conv2)
    conv2 = Conv2D(
        64, (3, 3), activation="relu", padding="same", kernel_initializer=HeNormal()
    )(conv2)
    conv2 = BatchNormalization()(conv2)
    conv2 = LeakyReLU(alpha=0.1)(conv2)
    conv2 = Dropout(dropout_rate)(conv2)

    # Decoder
    up1 = UpSampling2D(size=(2, 2))(conv2)
    concat1 = Concatenate()([up1, conv1])
    conv3 = Conv2D(
        32, (3, 3), activation="relu", padding="same", kernel_initializer=HeNormal()
    )(concat1)
    conv3 = BatchNormalization()(conv3)
    conv3 = LeakyReLU(alpha=0.1)(conv3)
    conv3 = Conv2D(
        32, (3, 3), activation="relu", padding="same", kernel_initializer=HeNormal()
    )(conv3)
    conv3 = BatchNormalization()(conv3)
    conv3 = LeakyReLU(alpha=0.1)(conv3)

    conv4 = Conv2D(1, (1, 1), activation="sigmoid")(conv3)

    model = Model(inputs=inputs, outputs=conv4)

    return model


nn_model = barcode_model()
