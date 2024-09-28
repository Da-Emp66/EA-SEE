import argparse
import cv2
import numpy as np
import os
import torch
import torch.nn as nn
import warnings

from dotenv import load_dotenv
from torch.optim.nadam import NAdam
from torch.utils.data import DataLoader
from typing import Union

from ea_see.dataset import FaceRecognitionDataset
from ea_see.model import FaceEmbeddingModel, FaceRecognitionModel

load_dotenv()

class FaceRecognizer:
    def __init__(
        self,
        weights_file: os.PathLike = os.getenv('WEIGHTS_FILE'),
        device: torch.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
    ):
        self.device = device

        self.weights_file = weights_file
        self.embedding_model = FaceEmbeddingModel(device)

        state = torch.load(self.weights_file)
        self.embedding_model.load_state_dict(state)

        self.classifier_model = FaceRecognitionModel(device)

    def train(
        self,
        dataset_path: os.PathLike = os.getenv('DATASET_DIR'),
        save_file: os.PathLike = "trained_weights.pt",
        train_validiation_ratio = 80 / 20,
        num_epochs: int = 100,
        learning_rate: float = 0.002,
    ):
        dataset = FaceRecognitionDataset(dataset_path)
        dataset_length = len(dataset)

        train_dataset_length = (train_validiation_ratio / (train_validiation_ratio + 1)) * dataset_length
        validation_dataset_length = dataset_length - train_dataset_length
        
        train_dataloader = DataLoader(dataset[:train_dataset_length], shuffle=True)
        validation_dataloader = DataLoader(dataset[train_dataset_length:], shuffle=True)

        loss_function = nn.CrossEntropyLoss()
        optimizer = NAdam(loss_function.parameters(), lr=learning_rate)

        best_valid_accuracy = -1 * float('inf')

        for epoch in range(num_epochs):
            self.classifier_model.train()

            # Train Loop
            for idx, (image, labels) in enumerate(train_dataloader):
                image = image.to(self.device)
                labels = labels.to(self.device)

                optimizer.zero_grad()

                predictions = self.classifier_model(image)
                loss = loss_function(predictions, labels)
                loss.backward()
                optimizer.step()

            self.classifier_model.eval()
            valid_accuracy = 0.0
            correct_labels, total_labels = 0, 0

            # Validation Loop
            for idx, (image, labels) in enumerate(validation_dataloader):
                image = image.to(self.device)
                labels = labels.to(self.device)

                optimizer.zero_grad()
                
                predictions = self.classifier_model(image)
                _, predicted = torch.max(predictions, 1)
                correct_labels += (predicted == labels).sum()
                total_labels += labels.shape[0]

            valid_accuracy = correct_labels * 100.0 / total_labels

            if valid_accuracy >= best_valid_accuracy:
                best_valid_accuracy = valid_accuracy
                torch.save(save_file)

    def __call__(self, image: Union[np.ndarray, os.PathLike]):
        if isinstance(image, str):
            image_matrix = cv2.imread(image)

        image_matrix = cv2.resize(image_matrix, (224, 224))
        prediction = self.model(image_matrix)

        return prediction

def main(args):
    fr = FaceRecognizer()

    if args.train:
        fr.train(dataset_path=args.dataset_dir)
    else:
        if args.image is not None:
            fr(args.image)
        else:
            warnings.warn("Cannot infer on an unspecified image. Use `--image` to pass image path.")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", dest="train", action="store_true", required=False)
    parser.add_argument("--dataset-dir", type=str, help="Path to dataset directory to train upon", default=os.getenv('DATASET_DIR'), required=False)
    parser.add_argument("--infer", dest="train", action="store_false", required=False)
    parser.add_argument("--image", type=str, help="Path to image to infer upon", default=None, required=False)
    args = parser.parse_args()
    main(args)
