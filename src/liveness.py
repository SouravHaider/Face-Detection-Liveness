"""Liveness / anti-spoofing.

Primary method: active challenge-response blink detection using the
Eye-Aspect-Ratio (EAR) from MediaPipe Face Mesh landmarks. A printed photo or
static replay cannot blink on demand, so requiring N blinks within a window
defeats the most common presentation attacks.

The class exposes a frame-by-frame API so it can drive either the notebook
webcam loop or the Streamlit app. A hook (`passive_score`) is provided to plug
in a texture/CNN passive anti-spoof model (e.g. MiniFASNet) later.
"""
from collections import deque

import numpy as np

from . import config

# MediaPipe Face Mesh indices for the eye landmarks used in the EAR formula.
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]


def _ear(pts):
    """Eye Aspect Ratio for 6 landmark points (p1..p6)."""
    p1, p2, p3, p4, p5, p6 = pts
    a = np.linalg.norm(p2 - p6)
    b = np.linalg.norm(p3 - p5)
    c = np.linalg.norm(p1 - p4)
    return (a + b) / (2.0 * c) if c > 0 else 0.0


class LivenessChecker:
    def __init__(self, ear_thr=config.EAR_THRESHOLD,
                 consec=config.EAR_CONSEC_FRAMES,
                 blinks_required=config.BLINKS_REQUIRED):
        self.ear_thr = ear_thr
        self.consec = consec
        self.blinks_required = blinks_required
        self._closed_frames = 0
        self.blinks = 0
        self.ear_history = deque(maxlen=64)
        try:
            import mediapipe as mp
            self._mesh = mp.solutions.face_mesh.FaceMesh(
                max_num_faces=1, refine_landmarks=True,
                min_detection_confidence=0.5, min_tracking_confidence=0.5)
        except Exception:
            self._mesh = None  # graceful: caller can fall back

    @property
    def is_live(self):
        return self.blinks >= self.blinks_required

    def reset(self):
        self._closed_frames = 0
        self.blinks = 0
        self.ear_history.clear()

    def update(self, frame_rgb):
        """Process one RGB frame; return dict with current state."""
        if self._mesh is None:
            return {"ear": None, "blinks": self.blinks, "is_live": False,
                    "error": "mediapipe not installed"}
        h, w = frame_rgb.shape[:2]
        res = self._mesh.process(frame_rgb)
        if not res.multi_face_landmarks:
            return {"ear": None, "blinks": self.blinks, "is_live": self.is_live}
        lm = res.multi_face_landmarks[0].landmark
        pts = np.array([[p.x * w, p.y * h] for p in lm])
        ear = (_ear(pts[LEFT_EYE]) + _ear(pts[RIGHT_EYE])) / 2.0
        self.ear_history.append(ear)

        if ear < self.ear_thr:
            self._closed_frames += 1
        else:
            if self._closed_frames >= self.consec:
                self.blinks += 1
            self._closed_frames = 0
        return {"ear": ear, "blinks": self.blinks, "is_live": self.is_live}

    @staticmethod
    def passive_score(face_crop):
        """Placeholder for a passive CNN anti-spoof model (e.g. MiniFASNet).
        Return a real/spoof probability in [0, 1]. Wire a trained model here."""
        raise NotImplementedError(
            "Plug a passive anti-spoof model (MiniFASNet/Silent-Face) here.")
