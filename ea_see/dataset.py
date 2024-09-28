import argparse
import cv2
import dlib
import os

from dotenv import load_dotenv
from torch.utils.data import Dataset

load_dotenv()

class FaceRecognitionDataset(Dataset):
    def __init__(self):
        pass

    def fabricate(
        self,
        model: os.PathLike,
        image_dir: os.PathLike,
        dataset: os.PathLike = os.getenv('DATASET_DIR'),
    ):
        prebuilt_face_detector = dlib.cnn_face_detection_model_v1("mmod_human_face_detector.dat")
        img = cv2.imread('path_to_image')
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        rects = prebuilt_face_detector(gray, 1)
        left, top, right, bottom = 0, 0, 0, 0

        for rect in rects:
            left = rect.rect.left()
            top = rect.rect.top()
            right = rect.rect.right()
            bottom = rect.rect.bottom()
            width = right - left
            height = bottom - top

            img_crop = img[top:top+height, left:left+width]
            cv2.imwrite('path_to_image_as_person_name', img_crop)

def main(args):
    dataset = FaceRecognitionDataset()
    dataset.fabricate(
        model=args.model,
        image_dir=args.image_dir,
        dataset=args.dataset
    )
    print(f"Dataset fabricated from `{args.image_dir}` in `{args.dataset}`.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="Path to model to use for dataset fabrication")
    parser.add_argument("--image-dir", type=str, help="Path to image directory to use for dataset fabrication")
    parser.add_argument("--dataset", type=str, help="Path to dataset directory to be fabricated", default=os.getenv('DATASET_DIR'))
    args = parser.parse_args()
    main(args)
