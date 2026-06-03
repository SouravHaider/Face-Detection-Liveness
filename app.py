"""Gradio web app: live face detection + blink-based liveness.

Designed for Hugging Face Spaces (SDK: Gradio, CPU Basic, free).
Run locally:  python app.py
"""
import cv2
import numpy as np
import gradio as gr

from src.liveness import LivenessChecker
from src import config

# Per-session liveness state is held in gr.State so multiple visitors don't clash.

def _load_detector():
    try:
        import tensorflow as tf
        if config.DETECTOR_WEIGHTS.exists():
            return tf.keras.models.load_model(config.DETECTOR_WEIGHTS, compile=False)
    except Exception:
        pass
    return None

DETECTOR = _load_detector()


def process(frame, checker):
    """frame: RGB numpy array from the streaming webcam."""
    if checker is None:
        checker = LivenessChecker()
    if frame is None:
        return frame, checker, "Waiting for camera…"

    img = frame.copy()
    state = checker.update(img)  # frame is already RGB in Gradio

    # Optional bounding box from the trained detector
    if DETECTOR is not None:
        inp = cv2.resize(img, config.IMG_SIZE) / 255.0
        cls, box = DETECTOR.predict(inp[None, ...], verbose=0)
        if cls[0][0] > 0.5:
            h, w = img.shape[:2]
            x1, y1, x2, y2 = (np.array(box[0]) * [w, h, w, h]).astype(int)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 200, 0), 2)

    live = state.get("is_live")
    color = (0, 200, 0) if live else (255, 0, 0)
    label = "LIVE ✅" if live else f"Blink {state.get('blinks', 0)}/{config.BLINKS_REQUIRED}"
    cv2.putText(img, label, (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
    if state.get("ear") is not None:
        cv2.putText(img, f"EAR {state['ear']:.2f}", (12, 64),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 180, 255), 1)

    status = ("✅ Liveness confirmed" if live
              else f"Blink to prove you're live — {state.get('blinks',0)}/{config.BLINKS_REQUIRED}")
    return img, checker, status


def reset():
    return LivenessChecker(), "Reset — blink to verify."


with gr.Blocks(title="Face Detection & Liveness") as demo:
    gr.Markdown(
        "# Face Detection & Liveness System\n"
        f"Real-time face tracking with **blink-based anti-spoofing**. "
        f"Blink {config.BLINKS_REQUIRED}× to prove you are a live person — "
        "a printed photo or video replay can't.")
    checker_state = gr.State(None)
    with gr.Row():
        cam = gr.Image(sources=["webcam"], streaming=True, label="Camera")
        out = gr.Image(label="Detection + Liveness")
    status = gr.Textbox(label="Status", interactive=False)
    reset_btn = gr.Button("Reset liveness check")

    cam.stream(process, inputs=[cam, checker_state],
               outputs=[out, checker_state, status])
    reset_btn.click(reset, outputs=[checker_state, status])

    with gr.Accordion("How liveness works", open=False):
        gr.Markdown(
            "Tracks your **Eye Aspect Ratio (EAR)** from MediaPipe face-mesh "
            "landmarks; a sharp EAR drop is a blink. Anti-spoofing quality is "
            "measured with APCER / BPCER / ACER (ISO/IEC 30107-3).")

if __name__ == "__main__":
    demo.launch()
