"""Evaluate the trained detector on the test set. Run: python -m src.evaluate"""
import numpy as np
import tensorflow as tf

from . import config, data, metrics


def main():
    model = tf.keras.models.load_model(
        config.DETECTOR_WEIGHTS, compile=False)
    _, test_ds, _ = data.load_datasets()

    pred_boxes, true_boxes, y_true_cls, y_pred_cls = [], [], [], []
    for X, y in test_ds:
        cls, box = model.predict(X, verbose=0)
        pred_boxes.extend(box.tolist())
        true_boxes.extend(np.array(y[1]).tolist())
        y_true_cls.extend(np.array(y[0]).flatten().tolist())
        y_pred_cls.extend((cls.flatten() > 0.5).astype(int).tolist())

    p, r = metrics.detection_precision_recall(pred_boxes, true_boxes, thr=0.5)
    print(f"Mean IoU         : {metrics.mean_iou(pred_boxes, true_boxes):.3f}")
    print(f"Precision@0.5    : {p:.3f}")
    print(f"Recall@0.5       : {r:.3f}")
    print(f"Confusion (face) : {metrics.confusion(y_true_cls, y_pred_cls)}")


if __name__ == "__main__":
    main()
