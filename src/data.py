"""Cross-platform data pipeline: collection, augmentation, tf.data building."""
import json
import time
import uuid
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf

from . import config


def collect_images(n=15, camera=0, out_dir=None, delay=0.5):
    """Capture n frames from a webcam into data/images."""
    out_dir = Path(out_dir or (config.DATA_DIR / "images"))
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(camera)
    try:
        for i in range(n):
            ret, frame = cap.read()
            if not ret:
                break
            cv2.imwrite(str(out_dir / f"{uuid.uuid1()}.jpg"), frame)
            cv2.imshow("collect", frame)
            time.sleep(delay)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


def _augmentor():
    import albumentations as alb
    return alb.Compose(
        [alb.RandomCrop(width=450, height=450),
         alb.HorizontalFlip(p=0.5),
         alb.RandomBrightnessContrast(p=0.2),
         alb.RandomGamma(p=0.2),
         alb.RGBShift(p=0.2),
         alb.VerticalFlip(p=0.5)],
        bbox_params=alb.BboxParams(format="albumentations",
                                   label_fields=["class_labels"]))


def build_augmented(per_image=config.AUG_PER_IMAGE):
    """Read data/<part>/{images,labels}, write aug_data/<part>/{images,labels}."""
    aug = _augmentor()
    w, h = config.CAPTURE_RES
    for part in config.PARTITIONS:
        (config.AUG_DIR / part / "images").mkdir(parents=True, exist_ok=True)
        (config.AUG_DIR / part / "labels").mkdir(parents=True, exist_ok=True)
        img_dir = config.DATA_DIR / part / "images"
        for img_path in img_dir.glob("*.jpg"):
            img = cv2.imread(str(img_path))
            coords = [0, 0, 0.00001, 0.00001]
            lbl_path = config.DATA_DIR / part / "labels" / f"{img_path.stem}.json"
            present = lbl_path.exists()
            if present:
                lbl = json.loads(lbl_path.read_text())
                pts = lbl["shapes"][0]["points"]
                coords = list(np.divide(
                    [pts[0][0], pts[0][1], pts[1][0], pts[1][1]], [w, h, w, h]))
            for x in range(per_image):
                try:
                    a = aug(image=img, bboxes=[coords], class_labels=["face"])
                except Exception:
                    continue
                out_img = config.AUG_DIR / part / "images" / f"{img_path.stem}.{x}.jpg"
                cv2.imwrite(str(out_img), a["image"])
                ann = {"image": img_path.name}
                if present and len(a["bboxes"]):
                    ann["bbox"] = list(a["bboxes"][0])
                    ann["class"] = 1
                else:
                    ann["bbox"] = [0, 0, 0, 0]
                    ann["class"] = 0
                (config.AUG_DIR / part / "labels" / f"{img_path.stem}.{x}.json").write_text(
                    json.dumps(ann))


def _load_image(path):
    img = tf.io.decode_jpeg(tf.io.read_file(path))
    img = tf.image.resize(img, config.IMG_SIZE)
    return img / 255.0


def _load_label(path):
    lbl = json.loads(Path(path.numpy().decode()).read_text())
    return [lbl["class"]], lbl["bbox"]


def _dataset(part):
    img_glob = str(config.AUG_DIR / part / "images" / "*.jpg")
    lbl_glob = str(config.AUG_DIR / part / "labels" / "*.json")
    imgs = tf.data.Dataset.list_files(img_glob, shuffle=False).map(_load_image)
    lbls = tf.data.Dataset.list_files(lbl_glob, shuffle=False).map(
        lambda x: tf.py_function(_load_label, [x], [tf.uint8, tf.float16]))
    ds = tf.data.Dataset.zip((imgs, lbls)).shuffle(2000)
    return ds.batch(config.BATCH_SIZE).prefetch(4)


def load_datasets():
    return tuple(_dataset(p) for p in config.PARTITIONS)
