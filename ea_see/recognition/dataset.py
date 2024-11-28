import argparse
import os
import re
import time
import cv2
import dlib
import numpy as np
import torch
import torchvision
import yaml

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
        
        class_mappings_file = "class_mappings.yaml",
    ):
        """Small dataset curated and fabricated manually"""

        # Set the transform to apply to images
        self.transform = transform

        # Fabricate the dataset if specified
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
        
        # Set up the internal record of samples
        self.dataset = os.path.join(dataset, split)
        self.items: Union[np.array, List] = np.array([])
        self.lazy_loading = lazy_loading
        if not lazy_loading:
            # Preload all images into matrices
            self.items = self._load_fabricated_dataset(self.dataset)
        else:
            # Store image paths, and load images only
            # when they are directly referenced
            self.items = []
            self.classes = os.listdir(self.dataset)
            self.num_classes = len(self.classes)
            for classname in self.classes:
                images_in_class = os.listdir(os.path.join(self.dataset, classname))
                self.items += [os.path.join(self.dataset, classname, image_file) for image_file in images_in_class]
                
        # Write the loaded classes to the class mappings file
        self.class_mappings_file = class_mappings_file
        with open(self.class_mappings_file, "w") as mappings:
            yaml.dump({"mappings": list(self.classes)}, mappings)
            mappings.close()
        
    def _load_fabricated_dataset(
        self,
        dataset: os.PathLike,
    ):
        """Preload all images in the dataset"""

        image_matrices = []

        # Every folder is a class
        self.classes = os.listdir(dataset)
        self.num_classes = len(self.classes)

        # Load all images in each folder and set the target class as the folder name
        for classname in self.classes:
            images_in_class = os.listdir(os.path.join(dataset, classname))
            for image_file in images_in_class:
                image_mat = cv2.imread(os.path.join(dataset, classname, image_file))
                image_matrices.append((classname, image_mat))
        
        # Return the list of classnames to image matrices
        return image_matrices

    def _fabricate(
        self,
        model: os.PathLike,
        image_dir: os.PathLike,
        dataset: os.PathLike,
    ):
        """Fabricate the dataset based on the image_dir
        using Dlib's face detector model for cropping faces"""
        
        # Ensure we are using CUDA to make predictions
        # using Dlib's face detector
        print(dlib.DLIB_USE_CUDA)
        print(dlib.cuda.get_num_devices())

        # Instantiate the face detection model
        prebuilt_face_detector = dlib.cnn_face_detection_model_v1(model)

        # Make the outer dataset directory
        os.makedirs(dataset, exist_ok=True)
        
        # For all images of format {classname}_{idx}.[jpg|png|...]
        for image_file in os.listdir(image_dir):
            try:
                # Establish the classname
                person_name, image_number = re.findall(r"(\w+_)+(\d+)\..*", image_file)[0]
                person_name = person_name.strip("_")

                # Load the image
                img = cv2.imread(os.path.join(image_dir, image_file))
                infer = dlib.load_rgb_image(os.path.join(image_dir, image_file))

                # Inference and time the inference to ensure CUDA usage
                begin = time.time()
                rects = prebuilt_face_detector(infer, 0)
                end = time.time()

                # Print the number of faces detected in the image, ideally should be 1
                print(f"{os.path.join(image_dir, image_file)}: {len(rects)} face detected in {end - begin} seconds")

                # Crop the face
                left, top, right, bottom = 0, 0, 0, 0

                for rect in rects:
                    left = rect.rect.left()
                    top = rect.rect.top()
                    right = rect.rect.right()
                    bottom = rect.rect.bottom()
                    width = right - left
                    height = bottom - top

                    img_crop = img[top:top+height, left:left+width]

                    # Save the face to the classname folder as {idx}.jpg
                    os.makedirs(os.path.join(dataset, person_name), exist_ok=True)
                    cv2.imwrite(os.path.join(dataset, person_name, f"{image_number}.jpg"), img_crop)
            except Exception as err:
                print(err)

    def __getitem__(self, idx: int):
        if self.lazy_loading:
            # Get the label from the stored filepath
            image_path = self.items[idx]
            classname = os.path.dirname(image_path).split("/")[-1]
            # Convert the label to one-hot representation
            labels = torch.zeros(self.num_classes, dtype=torch.float32)
            labels[self.classes.index(classname)] = 1
            # Load and transform the image
            return self.transform(cv2.imread(image_path)), labels
        else:
            # Get the label and pre-loaded image matrix from the internal storage
            classname, image_mat = self.items[idx]
            # Convert the label to one-hot representation
            labels = torch.zeros(self.num_classes, dtype=torch.float32)
            labels[self.classes.index(classname)] = 1
            # Transform the image matrix
            return self.transform(image_mat), labels

    def __len__(self):
        return len(self.items)

def main(args):
    # Instantiate the dataset to fabricate it
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
