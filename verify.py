"""Compare training preprocessing, PyTorch, Python ONNX, and C++ ONNX."""
import argparse
from pathlib import Path
import re
import subprocess
import tempfile

import numpy as np
import onnxruntime as ort
from PIL import Image
import torch
from torch import nn
from torchvision import models, transforms

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('images', nargs='*', type=Path,
                        default=[ROOT / 'data/chest_xray/test/NORMAL/IM-0001-0001.jpeg'])
    parser.add_argument('--cpp', type=Path, default=ROOT / 'build/classify')
    args = parser.parse_args()
    model = models.mobilenet_v3_large(weights=None)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, 2)
    model.load_state_dict(torch.load(ROOT / 'mobilenet_v3_xray.pth', map_location='cpu', weights_only=True))
    model.eval()
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    session = ort.InferenceSession(str(ROOT / 'mobilenet_v3_xray.onnx'), providers=['CPUExecutionProvider'])
    for path in args.images:
        with Image.open(path) as img:
            tensor = transform(img.convert('RGB')).unsqueeze(0)
        with torch.inference_mode():
            pytorch_scores = model(tensor).numpy()
        onnx_scores = session.run(['output'], {'input': tensor.numpy()})[0]
        print(f'\nImage: {path}', flush=True)
        print('Python tensor mean:', tensor.mean().item(), 'stddev:', tensor.std(correction=0).item())
        print('PyTorch raw scores:', pytorch_scores)
        print('Python ONNX raw scores:', onnx_scores)
        with tempfile.TemporaryDirectory() as tmp:
            dump = Path(tmp) / 'tensor.bin'
            result = subprocess.run([str(args.cpp.resolve()), str(path.resolve()),
                                     str(ROOT / 'mobilenet_v3_xray.onnx'), str(dump)],
                                    check=True, capture_output=True, text=True)
            print(result.stdout, end='')
            cpp_tensor = np.fromfile(dump, dtype=np.float32).reshape(tensor.shape)
        match = re.search(r'Raw scores -> NORMAL: ([^,]+), PNEUMONIA: ([^\n]+)', result.stdout)
        if not match:
            raise RuntimeError('C++ output is missing raw scores')
        cpp_scores = np.array([[float(match[1]), float(match[2])]])
        print('Maximum tensor difference:', np.max(np.abs(cpp_tensor - tensor.numpy())))
        print('Maximum ONNX score difference:', np.max(np.abs(cpp_scores - onnx_scores)))
        np.testing.assert_allclose(cpp_tensor, tensor.numpy(), rtol=0, atol=1e-6)
        np.testing.assert_allclose(pytorch_scores, onnx_scores, rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(cpp_scores, onnx_scores, rtol=1e-4, atol=1e-4)
        print('PASS: preprocessing and inference agree.')


if __name__ == '__main__':
    main()
