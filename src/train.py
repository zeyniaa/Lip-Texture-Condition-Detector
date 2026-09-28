"""
Training pipeline script for Lip Texture Condition Classifier.
"""

import os
import argparse
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from src.data_preprocessing import create_data_generators, clean_corrupted_images
from src.model import build_lip_cnn, compile_model
from src.evaluate import plot_training_history, evaluate_test_set


def parse_args():
    parser = argparse.ArgumentParser(description="Train 2D-CNN for Lip Texture Condition Classification.")
    parser.add_argument("--data-dir", type=str, default="data/processed", help="Path to balanced dataset root directory.")
    parser.add_argument("--epochs", type=int, default=20, help="Maximum number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for training and evaluation.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for Adam optimizer.")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience.")
    parser.add_argument("--use-gap", action="store_true", help="Use GlobalAveragePooling2D instead of Flatten.")
    parser.add_argument("--output-model", type=str, default="models/model_klasifikasi_bibir.h5", help="Path to save trained weights.")
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"🚀 Initializing training pipeline with data directory: {args.data_dir}")

    # Validate directory
    if not os.path.isdir(args.data_dir):
        print(f"❌ Error: Data directory '{args.data_dir}' not found.")
        print("💡 Place balanced dataset in data/processed or pass via --data-dir.")
        return

    # Clean any corrupted files
    for split in ['train', 'val', 'test']:
        split_dir = os.path.join(args.data_dir, split)
        if os.path.isdir(split_dir):
            removed = clean_corrupted_images(split_dir)
            if removed > 0:
                print(f"🧹 Removed {removed} corrupted image(s) from {split}")

    # Build generators
    train_gen, val_gen, test_gen = create_data_generators(
        args.data_dir,
        target_size=(224, 224),
        batch_size=args.batch_size,
        color_mode='grayscale'
    )

    print(f"📌 Class indices mapping: {train_gen.class_indices}")

    # Build and compile model
    model = build_lip_cnn(
        input_shape=(224, 224, 1),
        num_classes=3,
        dropout_rate=0.5,
        use_gap=args.use_gap
    )
    compile_model(model, learning_rate=args.lr)
    model.summary()

    # Callbacks
    os.makedirs(os.path.dirname(args.output_model), exist_ok=True)
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=args.patience,
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            filepath=args.output_model,
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1
        )
    ]

    # Training
    print("\n⏳ Starting model training...")
    history = model.fit(
        train_gen,
        epochs=args.epochs,
        validation_data=val_gen,
        callbacks=callbacks
    )

    # Plot metrics
    history_plot_path = "reports/training_curves.png"
    plot_training_history(history, save_path=history_plot_path)
    print(f"📈 Training curves saved to {history_plot_path}")

    # Evaluate on test set
    print("\n📊 Evaluating model on unseen test set...")
    cm_plot_path = "reports/confusion_matrix.png"
    results = evaluate_test_set(
        model,
        test_gen,
        save_cm_path=cm_plot_path
    )

    print(f"🎯 Test Accuracy: {results['test_accuracy']:.4f} | Test Loss: {results['test_loss']:.4f}")
    print("\n📋 Classification Report:")
    print(results['classification_report'])
    print(f"✅ Training and evaluation finished. Best model saved to {args.output_model}")


if __name__ == "__main__":
    main()
