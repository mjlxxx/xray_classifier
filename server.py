from pathlib import Path
from fastapi.responses import FileResponse
from fastapi import FastAPI, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

from inference import predict

app = FastAPI()

ROOT = Path(__file__).resolve().parent

@app.get("/")
def home():
    return FileResponse(ROOT / "index.html")


@app.post("/predict")
def predict_image(file: UploadFile):
    try:
        with Image.open(file.file) as image:
            image.load()
            result = predict(image)

        return result

    except (UnidentifiedImageError, OSError):
        raise HTTPException(
            status_code=400,
            detail="Could not read this image. Try a JPEG or PNG.",
        )

    finally:
        file.file.close()