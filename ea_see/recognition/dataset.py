import argparse
import os
import re
import time
import cv2
import dlib
import numpy as np
import torch
import torchvision

from dotenv import load_dotenv
from torch.utils.data import Dataset
from typing import List, Literal, Optional, Union

load_dotenv()

class CustomFaceRecognitionDataset(Dataset):
    def __init__(
        self,
        dataset: Optional[os.PathLike] = None,
        split: Literal["train", "valid"] = "train",
        transform: torchvision.transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Resize((224,224)),
            torchvision.transforms.Normalize((0.1307,), (0.3081,))
        ]),
        lazy_loading: bool = False,

        model_to_fabricate_with: Optional[os.PathLike] = None,
        image_dir_to_fabrication_from: Optional[os.PathLike] = None,
        dataset_dir_to_fabricate: Optional[os.PathLike] = None,
    ):
        self.transform = transform

        if dataset is None and \
            model_to_fabricate_with is not None and \
            image_dir_to_fabrication_from is not None and \
            dataset_dir_to_fabricate is not None:
            self._fabricate(
                model=model_to_fabricate_with,
                image_dir=image_dir_to_fabrication_from,
                dataset=dataset_dir_to_fabricate,
            )
            dataset = dataset_dir_to_fabricate
        elif dataset is None:
            raise ValueError("Must specify dataset path `dataset` if not fabricating.")
        
        self.dataset = os.path.join(dataset, split)
        self.items: Union[np.array, List] = np.array([])
        self.lazy_loading = lazy_loading
        if not lazy_loading:
            self.items = self._load_fabricated_dataset(self.dataset)
        else:
            self.items = []
            self.classes = os.listdir(self.dataset)
            self.num_classes = len(self.classes)
            for classname in self.classes:
                images_in_class = os.listdir(os.path.join(self.dataset, classname))
                self.items += [os.path.join(self.dataset, classname, image_file) for image_file in images_in_class]
        
    def _load_fabricated_dataset(
        self,
        dataset: os.PathLike,
    ):
        image_matrices = []

        self.classes = os.listdir(dataset)
        self.num_classes = len(self.classes)
        for classname in self.classes:
            images_in_class = os.listdir(os.path.join(dataset, classname))
            for image_file in images_in_class:
                image_mat = cv2.imread(os.path.join(dataset, classname, image_file))
                image_matrices.append((classname, image_mat))
        
        return image_matrices

    def _fabricate(
        self,
        model: os.PathLike,
        image_dir: os.PathLike,
        dataset: os.PathLike,
    ):
        print(dlib.DLIB_USE_CUDA)
        print(dlib.cuda.get_num_devices())
        prebuilt_face_detector = dlib.cnn_face_detection_model_v1(model)
        os.makedirs(dataset, exist_ok=True)
        
        for image_file in os.listdir(image_dir):
            try:
                person_name, image_number = re.findall(r"(\w+_)+(\d+)\..*", image_file)[0]
                person_name = person_name.strip("_")
                img = cv2.imread(os.path.join(image_dir, image_file))
                infer = dlib.load_rgb_image(os.path.join(image_dir, image_file))
                begin = time.time()
                rects = prebuilt_face_detector(infer, 0)
                end = time.time()
                print(f"{os.path.join(image_dir, image_file)}: {len(rects)} face detected in {end - begin} seconds")

                left, top, right, bottom = 0, 0, 0, 0

                for rect in rects:
                    left = rect.rect.left()
                    top = rect.rect.top()
                    right = rect.rect.right()
                    bottom = rect.rect.bottom()
                    width = right - left
                    height = bottom - top

                    img_crop = img[top:top+height, left:left+width]
                    os.makedirs(os.path.join(dataset, person_name), exist_ok=True)
                    cv2.imwrite(os.path.join(dataset, person_name, f"{image_number}.jpg"), img_crop)
            except Exception as err:
                print(err)

    def __getitem__(self, idx: int):
        if self.lazy_loading:
            image_path = self.items[idx]
            classname = os.path.dirname(image_path).split("/")[-1]
            labels = torch.zeros(self.num_classes, dtype=torch.float32)
            labels[self.classes.index(classname)] = 1
            return self.transform(cv2.imread(image_path)), labels
        else:
            classname, image_mat = self.items[idx]
            labels = torch.zeros(self.num_classes, dtype=torch.float32)
            labels[self.classes.index(classname)] = 1
            return self.transform(image_mat), labels

    def __len__(self):
        return len(self.items)

def main(args):
    _dataset = CustomFaceRecognitionDataset(
        model_to_fabricate_with=args.model,
        image_dir_to_fabrication_from=args.image_dir,
        dataset_dir_to_fabricate=args.dataset,
    )
    print(f"Dataset fabricated from `{args.image_dir}` in `{args.dataset}`.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="Path to model to use for dataset fabrication")
    parser.add_argument("--image-dir", type=str, help="Path to image directory to use for dataset fabrication")
    parser.add_argument("--dataset", type=str, help="Path to dataset directory to be fabricated")
    args = parser.parse_args()
    main(args)
