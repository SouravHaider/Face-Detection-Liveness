"""Dual-head face detector: VGG16 backbone -> (face? , bounding box)."""
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, GlobalMaxPooling2D
from tensorflow.keras.applications import VGG16

from . import config


def build_model(img_size=config.IMG_SIZE, freeze_backbone=True):
    """Build the dual-output detector.

    Returns a Keras Model with two heads:
      - classification head: sigmoid, face / no-face
      - regression head: 4 sigmoid outputs, bbox [x1, y1, x2, y2] (normalised)
    """
    inp = Input(shape=(*img_size, 3))
    backbone = VGG16(include_top=False)
    backbone.trainable = not freeze_backbone
    x = backbone(inp)

    # Classification head
    c = GlobalMaxPooling2D()(x)
    c = Dense(2048, activation="relu")(c)
    cls = Dense(1, activation="sigmoid", name="class")(c)

    # Regression head
    r = GlobalMaxPooling2D()(x)
    r = Dense(2048, activation="relu")(r)
    box = Dense(4, activation="sigmoid", name="box")(r)

    return Model(inputs=inp, outputs=[cls, box], name="facetracker")


def localization_loss(y_true, y_pred):
    """Penalise both corner position error and box size error."""
    delta_coord = tf.reduce_sum(tf.square(y_true[:, :2] - y_pred[:, :2]))
    h_true = y_true[:, 3] - y_true[:, 1]
    w_true = y_true[:, 2] - y_true[:, 0]
    h_pred = y_pred[:, 3] - y_pred[:, 1]
    w_pred = y_pred[:, 2] - y_pred[:, 0]
    delta_size = tf.reduce_sum(tf.square(w_true - w_pred) + tf.square(h_true - h_pred))
    return delta_coord + delta_size


class FaceTracker(Model):
    """Custom training loop combining classification + localization losses."""

    def __init__(self, detector, **kwargs):
        super().__init__(**kwargs)
        self.model = detector

    def compile(self, opt, classloss, localizationloss, **kwargs):
        super().compile(**kwargs)
        self.closs = classloss
        self.lloss = localizationloss
        self.opt = opt

    def _step(self, batch, training):
        X, y = batch
        classes, coords = self.model(X, training=training)
        closs = self.closs(y[0], classes)
        lloss = self.lloss(tf.cast(y[1], tf.float32), coords)
        total = lloss + config.CLASS_LOSS_WEIGHT * closs
        return total, closs, lloss

    def train_step(self, batch, **kwargs):
        with tf.GradientTape() as tape:
            total, closs, lloss = self._step(batch, training=True)
        grads = tape.gradient(total, self.model.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.model.trainable_variables))
        return {"total_loss": total, "class_loss": closs, "regress_loss": lloss}

    def test_step(self, batch, **kwargs):
        total, closs, lloss = self._step(batch, training=False)
        return {"total_loss": total, "class_loss": closs, "regress_loss": lloss}

    def call(self, X, **kwargs):
        return self.model(X, **kwargs)
