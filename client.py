import argparse
import os
import cv2
import numpy as np
import requests

from dotenv import load_dotenv
from typing import Union

load_dotenv()

SERVER_PORT = os.getenv('SERVER_PORT')

def remote_recognize(image: Union[np.ndarray, os.PathLike]):
    if isinstance(image, str):
        image = cv2.imread(image)
    
    _, image_buffer_arr = cv2.imencode(".jpg", image)
    image_bytes = image_buffer_arr.tobytes()

    response = requests.post(
        url=f"http://localhost:{SERVER_PORT}/recognize",
        json={ "image": image_bytes.hex() }
    )

    response_class = response.json()['prediction']
    return response_class

def recognize_faces_thread():
    pass

def capture_video_thread(device):
    capture = cv2.VideoCapture(device)

def main(args):
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("device", type=int, help="Video capture device to use for streaming")
    args = parser.parse_args()
    main(args)
