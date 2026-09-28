"""
Convolutional Neural Network (CNN) architectures for Lip Texture Condition Classification.
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout,
    GlobalAveragePooling2D,
    BatchNormalization
)
from typing import Tuple


def build_lip_cnn(
    input_shape: Tuple[int, int, int] = (224, 224, 1),
    num_classes: int = 3,
    dropout_rate: float = 0.5,
    use_gap: bool = False
) -> Sequential:
    """
    Build 2D-CNN for lip texture analysis.

    Args:
        input_shape: Dimensions of input tensor (height, width, channels).
                     Defaults to (224, 224, 1) grayscale.
        num_classes: Number of target categories (3: kurang_sehat, sehat, tidak_sehat).
        dropout_rate: Dropout regularization probability.
        use_gap: If True, uses GlobalAveragePooling2D instead of Flatten to drastically
                 reduce parameter size and prevent overfitting on small datasets.

    Returns:
        tf.keras.models.Sequential: Instantiated Keras model.
    """
    model = Sequential()
    model.add(Input(shape=input_shape))

    # Convolutional Block 1
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Convolutional Block 2
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Convolutional Block 3
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Head
    if use_gap:
        model.add(GlobalAveragePooling2D())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(dropout_rate))
    else:
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(dropout_rate))

    # Multi-class output probability layer
    model.add(Dense(num_classes, activation='softmax', name='classification_output'))

    return model


def compile_model(
    model: Sequential,
    learning_rate: float = 1e-3
) -> Sequential:
    """
    Compile model with Adam optimizer and Categorical Crossentropy loss.
    """
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model
