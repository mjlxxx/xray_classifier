# X-ray Classifier

A chest X-ray classification learning project built with **PyTorch, ONNX Runtime, C++, and FastAPI**. Upload an image or capture a photo of an existing X-ray, review the preview, and compare the model's NORMAL and PNEUMONIA scores.

This project explores the full path from transfer learning to a browser interface, including matching image preprocessing between Python and C++.

> Educational prototype, not a medical diagnostic tool. The model only predicts NORMAL or PNEUMONIA; it does not determine severity, subtype, or treatment. Softmax scores are not calibrated clinical confidence. Camera photographs have not been separately validated.

## Features

- JPEG and PNG uploads with image preview.
- Live camera preview, capture, stop, and retake by reopening the camera.
- Predictions with both class scores.
- MobileNetV3-Large training and ONNX export.
- A standalone C++ classifier using OpenCV and ONNX Runtime.
- A verification script comparing full input tensors and inference outputs across Python and C++.

## How it works

```text
Image upload / camera capture
           |
           v
Browser sends POST /predict
           |
           v
Pillow + torchvision preprocessing
           |
           v
ONNX Runtime -> raw scores -> softmax
           |
           v
Browser displays prediction and both class scores
```

The website uses Python inference. C++ is a separate command-line implementation, verified against the same model.

Preprocessing converts the image to RGB, produces three identical grayscale channels, resizes to 224 × 224 using antialiased bilinear filtering, scales pixels to [0, 1], and applies ImageNet normalization. The model receives a float32 tensor with shape `[1, 3, 224, 224]`.

## Run locally

### 1. Clone and install

The current dependency pins reflect the author's Python 3.12 environment on an Intel Mac. Other platforms may require compatible PyTorch builds; a fresh cross-platform installation has not been tested.

```bash
git clone https://github.com/mjlxxx/xray_classifier.git
cd xray_classifier
python3.12 -m venv venv
source venv/bin/activate
python -m pip install -r requirements.txt
```

### 2. Supply the model

**The dataset and trained model files are not included in this repository.** Before starting the server, place `mobilenet_v3_xray.onnx` in the project root, or follow the training instructions below to generate it. The server loads the model at startup and cannot run without it.

For verification against PyTorch, also provide the matching `mobilenet_v3_xray.pth` checkpoint. Both files must come from the same training run.

### 3. Start the website

```bash
python -m uvicorn server:app --reload
```

Open **http://127.0.0.1:8000/**. Choose an image, or open the camera and capture a photo, then click the circular **Analyze image** arrow.

Use the server URL rather than opening `index.html` as a file: the page sends requests to `/predict`. Camera use requires browser permission and a connected camera. Localhost is suitable for local development; remote camera access requires HTTPS.

Interactive API documentation is available at **http://127.0.0.1:8000/docs**. Stop the server with `Ctrl+C`.

## Command-line inference

```bash
python inference.py path/to/xray.jpeg
```

Example response for the author's local model and `IM-0001-0001.jpeg`:

```json
{
  "prediction": "NORMAL",
  "probabilities": {
    "NORMAL": 0.9508745074272156,
    "PNEUMONIA": 0.0491255559027195
  },
  "raw_scores": [1.6158537864685059, -1.3471490144729614]
}
```

Retraining can produce different scores. The raw scores are logits, not percentages.

## Training

Download the [Chest X-Ray Images (Pneumonia) dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) separately and consult its usage terms. Arrange the extracted files as:

```text
data/chest_xray/
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
├── val/
│   ├── NORMAL/
│   └── PNEUMONIA/
└── test/
    ├── NORMAL/
    └── PNEUMONIA/
```

From the project root:

```bash
python train.py
```

The script loads ImageNet-pretrained MobileNetV3-Large, freezes existing parameters, replaces the final classification layer, and trains for five epochs using weighted cross-entropy. It uses Apple MPS when available and otherwise CPU. The first run may download pretrained weights.

It reports validation and test accuracy, then writes `mobilenet_v3_xray.pth` and `mobilenet_v3_xray.onnx`, overwriting files with those names. Class weights currently use hardcoded training counts, so update them if you change the dataset. The training script does not fix random seeds; runs are not guaranteed to reproduce identical models.

## Build and verify C++ inference

Install CMake, OpenCV, and ONNX Runtime. On macOS with Homebrew:

```bash
brew install cmake opencv onnxruntime
cmake -S . -B build
cmake --build build
./build/classify data/chest_xray/test/NORMAL/IM-0001-0001.jpeg
```

`CMakeLists.txt` currently assumes Intel Homebrew paths under `/usr/local/opt/onnxruntime`. On Apple Silicon or other systems, adjust its include/library paths to your ONNX Runtime installation before configuring.

Run the cross-language comparison:

```bash
python verify.py
```

Or pass one or more images:

```bash
python verify.py path/to/first.jpeg path/to/second.png
```

The script checks full preprocessed tensors, PyTorch vs. Python ONNX scores, and C++ vs. Python ONNX scores. It prints `PASS: preprocessing and inference agree.` when checks pass.

### Why preprocessing parity matters

The original C++ pipeline used ordinary OpenCV bilinear resizing, whereas training used Pillow's antialiased resizing. On one tested image, this changed the predicted class even though tensor averages were close. `preprocess.h` implements grayscale conversion and separable antialiased bilinear resizing to match the Python pipeline.

During local checks on four X-rays and four synthetic PNGs, input tensors matched exactly, and Python/C++ ONNX score differences were below 0.000006. These are implementation parity checks, not a clinical validation or a full test-set evaluation. Image decoders and library versions can introduce differences on other inputs.

## Project files

| File | Purpose |
| --- | --- |
| `index.html` | Page layout, styling, upload, camera, and API requests |
| `server.py` | FastAPI homepage and prediction endpoint |
| `inference.py` | Reusable preprocessing and ONNX prediction function |
| `train.py` | Training, evaluation, checkpoint saving, and ONNX export |
| `verify.py` | Python/PyTorch/ONNX/C++ parity checks |
| `main.cpp` | C++ command-line inference |
| `preprocess.h` | Pillow-compatible grayscale and resize implementation |
| `CMakeLists.txt` | C++ build configuration |
| `requirements.txt` | Python dependency versions from the local environment |

## Current scope and next steps

The app is a local prototype. It has no authentication, explicit upload-size limits, or production deployment configuration. Do not treat making the source public as making the server production-ready.

Planned learning directions:

- Full test-set confusion matrix, precision, recall, and per-class errors.
- Original-file vs. camera-photo comparisons.
- Better result presentation and image metadata.
- Exploring heatmaps and their limitations.
- Deployment preparation after evaluation and input handling improvements.

## Credits and licensing

- Dataset: [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia).
- Interface styling inspired by [Flair Social](https://flair.social/).
- Built with PyTorch/torchvision, Pillow, NumPy, ONNX Runtime, OpenCV, and FastAPI.

No project license has been selected yet. Public availability alone does not grant an open-source license. Dependencies and datasets retain their own licenses and terms.
