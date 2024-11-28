import os
import cv2
import numpy as np
import re
from ordered_set import OrderedSet
import torch
import torchvision
import yaml

from dotenv import load_dotenv
from torch.utils.data import Dataset
from torchvision.transforms import v2 as v2
from typing import List, Literal, Optional, Tuple, Union

load_dotenv()

class LargerFaceRecognitionDataset(Dataset):
    def __init__(
        self,
        dataset: Optional[os.PathLike] = "archive",
        split: Literal["train", "valid"] = "train",
        transform: torchvision.transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Resize((224,224)),
            torchvision.transforms.Normalize((0.1307,), (0.3081,))
        ]),
        lazy_loading: bool = False,
        class_mappings_file = "class_mappings.yaml",
    ):
        """Larger dataset available on Kaggle"""

        # Set transform and split
        self.transform = transform
        self.split = split

        # Instantiate the internal storage
        self.dataset = os.path.join(dataset, "Faces", "Faces")
        self.items: Union[np.array, List] = np.array([])
        self.lazy_loading = lazy_loading
        if not lazy_loading:
            # Store the pre-loaded image matrices and classes
            self.items = self._load_fabricated_dataset(self.dataset)
            self.classes = OrderedSet([classname for classname, _ in self.items])
        else:
            # Store the image paths and classes
            self.items = os.listdir(self.dataset)
            self.classes = OrderedSet([re.findall("((?:\w+\s)*\w+)_\d+\.jpg", sample)[0] for sample in self.items])

        # Choose whether this is the train split or validation split
        train, valid = self._train_valid_split(self.items, self.classes)
        if self.split == "train":
            self.items = train
        else:
            self.items = valid

        # Dump the class mappings to the class mappings file
        self.num_classes = len(self.classes)
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

        samples = os.listdir(dataset)
        for sample in samples:
            # Load the image and obtain the classname
            # from a regular expression
            image_mat = cv2.imread(os.path.join(dataset, sample))
            classname = re.findall("((?:\w+\s)*\w+)_\d+\.jpg", sample)[0]
            image_matrices.append((classname, image_mat))
        
        return image_matrices
    
    def _train_valid_split(
            self,
            items: List[Union[Tuple[str, np.array], str]],
            classes: List[str],
            ratio_valid: Optional[float] = 0.25,
        ):
        """Split the dataset into train and valid sets"""

        # Instantiate accumulators
        train = []
        valid = []
        ratio_train = 1.0 - ratio_valid

        number_of_samples = {classname: 0 for classname in classes}

        # If lazy-loaded
        if type(items[0]) == str:
            # Count up the total number of images of each class
            for sample in items:
                for classname in number_of_samples:
                    if classname in sample:
                        number_of_samples[classname] += 1

            # Distribute the images for each class
            # based on the train-valid balance ratio
            for sample in items:
                print(sample)
                sample_idx = int(re.findall("(?:(?:\w+\s)*\w+)_(\d+)\.jpg", sample)[0])
                classname = re.findall("((?:\w+\s)*\w+)_\d+\.jpg", sample)[0]
                # If we are over the number of samples in the train ratio,
                if sample_idx > number_of_samples[classname] * ratio_train:
                    # append to validation dataset.
                    valid.append(sample)
                else:
                    # Otherwise, append to train dataset.
                    train.append(sample)

        elif type(items[0]) == tuple:
            # Preloaded

            # Count the total number of samples per class
            for sample in items:
                classname, _ = sample
                number_of_samples[classname] += 1
            
            counters = {classname: 0 for classname in number_of_samples}

            # For all items in the internal storage
            for sample in items:
                sample_idx = counters[classname]
                classname, _ = sample
                # If we are over the number of samples in the train ratio,
                if sample_idx > number_of_samples[classname] * ratio_train:
                    # append to validation dataset.
                    valid.append(sample)
                else:
                    # Otherwise, append to train dataset.
                    train.append(sample)
                counters[classname] += 1
        else:
            raise TypeError("Got an unsupported type for `items`.")

        return train, valid

    def __getitem__(self, idx: int):
        if self.lazy_loading:
            # Get the filename and classname
            image_path = self.items[idx]
            classname = re.findall("((?:\w+\s)*\w+)_\d+\.jpg", image_path)[0]
            # Convert labels to one-hot based on class mappings
            labels = torch.zeros(self.num_classes, dtype=torch.float32)
            labels[self.classes.index(classname)] = 1
            # Load the image and apply transform
            return self.transform(cv2.imread(image_path)), labels
        else:
            # Get the classname and image matrix
            classname, image_mat = self.items[idx]
            # Convert labels to one-hot based on class mappings
            labels = torch.zeros(self.num_classes, dtype=torch.float32)
            labels[self.classes.index(classname)] = 1
            # Apply transform to the image matrix
            return self.transform(image_mat), labels

    def __len__(self):
        return len(self.items)

def main():
    # Instantiate the dataset as a test
    _dataset = LargerFaceRecognitionDataset()

if __name__ == "__main__":
    main()
