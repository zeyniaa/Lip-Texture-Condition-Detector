"""
Gradio Interactive Web Application for Lip Texture Condition Classification.
Supports real-time image upload, grayscale texture preprocessing, and multi-class probability scoring.
"""

import os
from typing import Tuple
import numpy as np
import tensorflow as tf
from PIL import Image
import gradio as gr

from src.model import build_lip_cnn


MODEL_PATH = os.environ.get("MODEL_PATH", "models/model_klasifikasi_bibir.h5")
CLASS_NAMES = ['kurang_sehat', 'sehat', 'tidak_sehat']
CLASS_DESCRIPTIONS = {
    'sehat': "✅ Bibir Sehat (Normal, terhidrasi dengan baik, tanpa luka atau fisur).",
    'kurang_sehat': "⚠️ Kurang Sehat (Kering, dehidrasi ringan, pecah-pecah permukaan).",
    'tidak_sehat': "⛔ Tidak Sehat (Luka terbuka, fisur dalam, peradangan, atau lesi parah)."
}

# Load model or create initialized instance
_model = None


def get_model():
    global _model
    if _model is not None:
        return _model

    if os.path.exists(MODEL_PATH):
        try:
            print(f"📦 Loading pre-trained weights from {MODEL_PATH}")
            _model = tf.keras.models.load_model(MODEL_PATH)
            return _model
        except Exception as e:
            print(f"⚠️ Failed to load model weights from {MODEL_PATH}: {e}")

    print("ℹ️ Weights file not found locally. Instantiating architecture for demo/inference.")
    _model = build_lip_cnn(input_shape=(224, 224, 1), num_classes=3)
    return _model


def predict_lip_condition(input_image) -> Tuple[str, str]:
    """
    Inference handler for Gradio UI.
    Accepts filepath or PIL Image, normalizes to grayscale 224x224, and predicts class probabilities.
    """
    if input_image is None:
        return "⚠️ Silakan unggah gambar bibir terlebih dahulu.", None

    # Load and preprocess
    if isinstance(input_image, str):
        img = Image.open(input_image).convert('L')
    else:
        img = Image.fromarray(input_image).convert('L')

    img_resized = img.resize((224, 224), Image.Resampling.NEAREST)
    img_array = np.array(img_resized, dtype=np.float32) / 255.0
    img_tensor = np.expand_dims(img_array, axis=(0, -1))  # (1, 224, 224, 1)

    model = get_model()
    pred_probs = model.predict(img_tensor, verbose=0)[0]

    pred_idx = int(np.argmax(pred_probs))
    pred_label = CLASS_NAMES[pred_idx]
    confidence = float(pred_probs[pred_idx])

    desc = CLASS_DESCRIPTIONS.get(pred_label, "")

    result_text = (
        f"### 📋 Hasil Klasifikasi\n"
        f"- **Prediksi Kategori**: `{pred_label.upper()}`\n"
        f"- **Confidence Score**: `{confidence * 100:.2f}%`\n\n"
        f"**Interpretasi Klinis / Estetika:**\n{desc}\n\n"
        f"**Distribusi Probabilitas:**\n"
    )
    for c_name, prob in zip(CLASS_NAMES, pred_probs):
        bar_len = int(prob * 20)
        bar = "█" * bar_len + "░" * (20 - bar_len)
        result_text += f"- `{c_name:<13}` : `[{bar}]` **{prob * 100:.2f}%**\n"

    if not os.path.exists(MODEL_PATH):
        result_text += "\n> 💡 *Catatan: Model bobot tersimpan (`models/model_klasifikasi_bibir.h5`) belum diunduh. Inferensi saat ini menggunakan arsitektur baseline.*"

    return result_text, img_resized


def create_ui():
    demo = gr.Interface(
        fn=predict_lip_condition,
        inputs=gr.Image(
            label="📷 Unggah Foto Bibir (Warna atau Grayscale)",
            type="filepath"
        ),
        outputs=[
            gr.Markdown(label="📋 Analisis & Probabilitas"),
            gr.Image(label="🖼️ Citra Pra-pemrosesan (224x224 Grayscale)")
        ],
        title="👄 Lip Texture Condition Detector",
        description=(
            "Sistem klasifikasi kondisi kesehatan bibir berbasis **Deep Learning (2D-CNN)** "
            "dengan analisis tekstur citra grayscale ke dalam 3 kategori: **Sehat**, **Kurang Sehat**, dan **Tidak Sehat**."
        ),
        examples=[],
        allow_flagging="never"
    )
    return demo


if __name__ == "__main__":
    app = create_ui()
    app.launch(share=False)
