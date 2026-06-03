# Deploying to Hugging Face Spaces (free)

HF Spaces now offers **Gradio, Docker, or Static** as SDKs (Streamlit is no
longer a one-click option). This app is built with **Gradio**.

## 1. Create the Space
Go to https://huggingface.co/new-space and choose:
- SDK: **Gradio**, template **Blank**
- Hardware: **CPU Basic (Free)**
- Visibility: **Public**

The build config Gradio needs is already in the front-matter at the top of
`README.md` (`sdk: gradio`, `app_file: app.py`), so it will build automatically.

## 2. Push your code
The Space is a git repo:

```bash
cd "/path/to/Final Year Projects - COMP 1682"
git init
git add .
git commit -m "Face detection + liveness Gradio app"
git remote add space https://huggingface.co/spaces/<your-username>/<space-name>
git push space main
```

When prompted for a password, paste a **Write access token** from
https://huggingface.co/settings/tokens (not your account password).

## 3. Finish
- HF installs `requirements.txt` and runs `app.py` (Gradio).
- The webcam uses the visitor's browser, so anyone with the link can try it.
- Copy the Space URL into the README "Try it live" line.

> Free CPU runs the TensorFlow detector slowly; the blink/liveness check stays
> fast. To skip the detector, the app simply shows liveness only when no
> trained `models/facetracker.h5` is present.
