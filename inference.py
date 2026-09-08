from pathlib import Path

import onnxruntime as ort
import torch
from PIL import Image
from torchvision import transforms

ROOT = Path(__file__).resolve().parent
LABELS = ["NORMAL", "PNEUMONIA"]

session = ort.InferenceSession(
    str(ROOT / "mobilenet_v3_xray.onnx"),
    providers=["CPUExecutionProvider"],
)

preprocess = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


def predict(image: Image.Image) -> dict:
    tensor = preprocess(image.convert("RGB")).unsqueeze(0)

    logits = session.run(
        ["output"],
        {"input": tensor.numpy()},
    )[0][0]

    probabilities = torch.softmax(
        torch.from_numpy(logits), dim=0
    ).tolist()

    best_index = int(logits.argmax())

    return {
        "prediction": LABELS[best_index],
        "probabilities": dict(zip(LABELS, probabilities)),
        "raw_scores": logits.tolist(),
    }

if __name__ == "__main__":
    import json
    import sys

    with Image.open(sys.argv[1]) as image:
        result = predict(image)

    print(json.dumps(result, indent=2))