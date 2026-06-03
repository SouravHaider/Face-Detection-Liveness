"""Train the detector and save weights. Run: python -m src.train"""
import tensorflow as tf

from . import config, data
from .model import build_model, FaceTracker, localization_loss


def main():
    for gpu in tf.config.experimental.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)

    train_ds, _, val_ds = data.load_datasets()
    batches = max(1, len(train_ds))
    lr_decay = (1.0 / 0.75 - 1) / batches
    try:
        opt = tf.keras.optimizers.Adam(learning_rate=config.LEARNING_RATE, weight_decay=lr_decay)
    except TypeError:  # older TF
        opt = tf.keras.optimizers.Adam(learning_rate=config.LEARNING_RATE)

    model = FaceTracker(build_model())
    model.compile(opt, tf.keras.losses.BinaryCrossentropy(), localization_loss)

    cbs = [
        tf.keras.callbacks.TensorBoard(log_dir=str(config.LOGS_DIR)),
        tf.keras.callbacks.EarlyStopping(monitor="val_total_loss", patience=6,
                                         restore_best_weights=True),
    ]
    model.fit(train_ds, epochs=config.EPOCHS, validation_data=val_ds, callbacks=cbs)
    model.model.save(config.DETECTOR_WEIGHTS)
    print(f"Saved detector to {config.DETECTOR_WEIGHTS}")


if __name__ == "__main__":
    main()
