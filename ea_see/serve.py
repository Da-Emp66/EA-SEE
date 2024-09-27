from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class ImageHex(BaseModel):
    image: str

@app.post("/recognize")
def inference(image: ImageHex):
    pass

