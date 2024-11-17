import os
import cv2
import numpy as np
import re
from ordered_set import OrderedSet
import torch
import torchvision

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
    ):
        self.transform = transform
        self.split = split

        self.dataset = os.path.join(dataset, "Faces", "Faces")
        self.items: Union[np.array, List] = np.array([])
        self.lazy_loading = lazy_loading
        if not lazy_loading:
            self.items = self._load_fabricated_dataset(self.dataset)
            self.classes = OrderedSet([classname for classname, _ in self.items])
        else:
            self.items = os.listdir(self.dataset)
            self.classes = OrderedSet([re.findall("((?:\w+\s)*\w+)_\d+\.jpg", sample)[0] for sample in self.items])

        train, valid = self._train_valid_split(self.items, self.classes)
        if self.split == "train":
            self.items = train
        else:
            self.items = valid

        self.num_classes = len(self.classes)
        
    def _load_fabricated_dataset(
        self,
        dataset: os.PathLike,
    ):
        image_matrices = []

        samples = os.listdir(dataset)
        for sample in samples:
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
        train = []
        valid = []
        ratio_train = 1.0 - ratio_valid

        number_of_samples = {classname: 0 for classname in classes}

        if type(items[0]) == str:
            for sample in items:
                for classname in number_of_samples:
                    if classname in sample:
                        number_of_samples[classname] += 1

            for sample in items:
                print(sample)
                sample_idx = int(re.findall("(?:(?:\w+\s)*\w+)_(\d+)\.jpg", sample)[0])
                classname = re.findall("((?:\w+\s)*\w+)_\d+\.jpg", sample)[0]
                if sample_idx > number_of_samples[classname] * ratio_train:
                    valid.append(sample)
                else:
                    train.append(sample)

        elif type(items[0]) == tuple:
            for sample in items:
                classname, _ = sample
                number_of_samples[classname] += 1
            
            counters = {classname: 0 for classname in number_of_samples}

            for sample in items:
                sample_idx = counters[classname]
                classname, _ = sample
                if sample_idx > number_of_samples[classname] * ratio_train:
                    valid.append(sample)
                else:
                    train.append(sample)
                counters[classname] += 1
        else:
            raise TypeError("Got an unsupported type for `items`.")

        return train, valid

    def __getitem__(self, idx: int):
        if self.lazy_loading:
            image_path = self.items[idx]
            classname = re.findall("((?:\w+\s)*\w+)_\d+\.jpg", image_path)[0]
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

def main():
    _dataset = LargerFaceRecognitionDataset()

if __name__ == "__main__":
    main()
