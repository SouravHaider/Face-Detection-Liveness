---
title: Face Detection And Liveness
emoji: 🟢
colorFrom: green
colorTo: blue
sdk: gradio
app_file: app.py
pinned: false
license: mit
---

# Face Detection & Liveness System

Real-time **face detection** with **blink-based liveness / anti-spoofing**. A VGG16
dual-head network locates the face and predicts a bounding box, while an
Eye-Aspect-Ratio (EAR) challenge-response layer verifies the face belongs to a
*live person* — not a printed photo or a video replay.

Originally built as a Final Year Project (COMP 1682) and since extended into a
deployable, evaluated, portfolio-grade system.

> **Try it live:** _add your Hugging Face Space URL here after deploying_
> **Demo GIF:** _record a short screen capture of `streamlit run app.py` and drop it here_

---

## Why this is more than a face detector

A bounding-box model alone will happily accept a photo held up to the camera.
This project adds a **presentation-attack defence**: the user must blink on
demand, which a static image or loop cannot do. Liveness quality is measured
with the standard ISO/IEC 30107-3 metrics — **APCER, BPCER, ACER**.

---

## Architecture

```
Input (120×120×3)
    └── VGG16 (frozen backbone, no top)
         ├── GlobalMaxPooling → Dense(2048, relu) → Dense(1, sigmoid)   ← face / no-face
         └── GlobalMaxPooling → Dense(2048, relu) → Dense(4, sigmoid)   ← bbox [x1,y1,x2,y2]

Liveness layer (MediaPipe Face Mesh)
    └── Eye Aspect Ratio → blink counter → live / spoof decision
```

A custom localization loss penalises both corner position error and box size error:

```python
total_loss = localization_loss + 0.5 * binary_cross_entropy
```

---

## Project structure

```
src/
  config.py      — central, cross-platform paths & hyperparameters
  data.py        — capture, Albumentations augmentation, tf.data pipeline
  model.py       — dual-head detector, custom loss, training loop
  liveness.py    — EAR blink-based anti-spoofing (MediaPipe)
  metrics.py     — IoU, precision/recall, APCER/BPCER/ACER
  train.py       — train & save the detector
  evaluate.py    — quantitative evaluation on the test set
app.py           — Streamlit + WebRTC live web app
demo/            — original notebook (FD.ipynb) + portfolio page
requirements.txt, LICENSE, .gitignore
```

---

## Quickstart

```bash
pip install -r requirements.txt

# 1. collect & label images (LabelMe), then build the augmented set
python -c "from src import data; data.collect_images(n=30)"
labelme                      # annotate, save JSON next to images
python -c "from src import data; data.build_augmented()"

# 2. train and evaluate
python -m src.train
python -m src.evaluate

# 3. run the web app (browser webcam, no OS access needed)
python app.py
```

---

## Results

| Metric | Value |
|---|---|
| Mean IoU (test) | _run `python -m src.evaluate`_ |
| Precision @ IoU 0.5 | _…_ |
| Recall @ IoU 0.5 | _…_ |
| ACER (liveness) | _…_ |

> Fill these in after training — quantitative results are what set a portfolio
> project apart from a tutorial.

---

## Roadmap

- [x] Cross-platform, modular codebase
- [x] Blink-based active liveness
- [x] Detection + anti-spoofing metrics
- [x] Deployable Streamlit/WebRTC app
- [ ] Passive CNN anti-spoof model (MiniFASNet / Silent-Face) via `LivenessChecker.passive_score`
- [ ] Train on a public face dataset (WIDER FACE) + spoof set (CelebA-Spoof / CASIA-FASD)
- [ ] ONNX export + quantization for real-time edge inference
- [ ] GitHub Actions CI

---

## Deployment (Hugging Face Spaces)

The repo ships with a Space config in `README` front-matter compatible files.
Create a new **Streamlit** Space, push this repo, and the app builds from
`requirements.txt` + `app.py` automatically. See `DEPLOY.md`.

## License

MIT — see [LICENSE](LICENSE).
