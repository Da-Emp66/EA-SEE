import os
import cv2
import numpy as np

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from ea_see.recognition.component import FaceRecognizer

app = FastAPI()
fr = FaceRecognizer(os.getenv('CLASSIFIER_WEIGHTS_FILE'))

class ImageHex(BaseModel):
    image: str

@app.post("/recognize")
def inference(image: ImageHex):
    try:
        nparray = np.frombuffer(bytes.fromhex(image['image']), np.uint8)
        decoded_image_matrix = cv2.imdecode(nparray, cv2.IMREAD_COLOR)
        prediction = fr(decoded_image_matrix)
        return { "prediction": prediction }
    except Exception as e:
        raise HTTPException(status_code=404, detail=e)
    