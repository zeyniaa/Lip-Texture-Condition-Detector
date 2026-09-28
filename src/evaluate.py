"""
Evaluation and diagnostic routines: learning curves, confusion matrix, and classification report.
"""

import os
from typing import List, Dict, Any
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf


def plot_training_history(history: Any, save_path: str = None) -> None:
    """
    Plot training vs validation accuracy and loss side-by-side.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Accuracy Plot
    axes[0].plot(history.history.get('accuracy', []), label='Train Accuracy', color='#2563eb')
    axes[0].plot(history.history.get('val_accuracy', []), label='Val Accuracy', color='#16a34a')
    axes[0].set_title('Accuracy per Epoch')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].grid(True, linestyle='--', alpha=0.5)
    axes[0].legend()

    # Loss Plot
    axes[1].plot(history.history.get('loss', []), label='Train Loss', color='#dc2626')
    axes[1].plot(history.history.get('val_loss', []), label='Val Loss', color='#d97706')
    axes[1].set_title('Loss per Epoch')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].grid(True, linestyle='--', alpha=0.5)
    axes[1].legend()

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
    plt.close()


def evaluate_test_set(
    model: tf.keras.Model,
    test_generator: Any,
    class_labels: List[str] = None,
    save_cm_path: str = None
) -> Dict[str, Any]:
    """
    Evaluate trained model on test generator and compute comprehensive metrics.
    """
    if class_labels is None:
        class_labels = list(test_generator.class_indices.keys())

    test_loss, test_acc = model.evaluate(test_generator, verbose=1)

    # Predictions
    y_pred_probs = model.predict(test_generator, verbose=1)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = test_generator.classes

    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=class_labels, output_dict=False)
    report_dict = classification_report(y_true, y_pred, target_names=class_labels, output_dict=True)

    # Plot Confusion Matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_labels,
        yticklabels=class_labels
    )
    plt.title('Confusion Matrix - Test Set')
    plt.xlabel('Predicted Label')
    plt.ylabel('Ground Truth')
    plt.tight_layout()

    if save_cm_path:
        os.makedirs(os.path.dirname(save_cm_path), exist_ok=True)
        plt.savefig(save_cm_path, dpi=300)
    plt.close()

    return {
        'test_loss': float(test_loss),
        'test_accuracy': float(test_acc),
        'confusion_matrix': cm.tolist(),
        'classification_report': report,
        'metrics_dict': report_dict
    }
