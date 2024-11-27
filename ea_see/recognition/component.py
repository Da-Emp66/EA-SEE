import argparse
import cv2
import numpy as np
import os
import torch
import torch.nn as nn
import torchvision
import warnings
import yaml

from dotenv import load_dotenv
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from typing import Any, Optional, Union

from ea_see.recognition.dataset import CustomFaceRecognitionDataset
from ea_see.recognition.larger_dataset import LargerFaceRecognitionDataset
from ea_see.recognition.model import FaceEmbeddingModel, FaceRecognitionModel

load_dotenv()

class FaceRecognizer:
    def __init__(
        self,
        classifier_weights_file: os.PathLike = None,
        embedding_weights_file: os.PathLike = os.getenv('EMBEDDING_WEIGHTS_FILE'),
        device: torch.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
        class_mappings_file = "class_mappings.yaml",
    ):
        self.device = device

        self.embedding_weights_file = embedding_weights_file
        self.embedding_model = FaceEmbeddingModel().to(device)

        state = torch.load(self.embedding_weights_file, weights_only=False)
        self.embedding_model.load_state_dict(state)
        self.embedding_model.compile()

        self.classifier_weights_file = classifier_weights_file
        self.classifier_model = FaceRecognitionModel(int(os.getenv('DATASET_CLASSES'))).to(device)
        
        if classifier_weights_file is not None:
            state = torch.load(self.classifier_weights_file, weights_only=False)
            self.classifier_model.load_state_dict(state)
        self.classifier_model.compile()

        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Resize((224,224)),
            torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            )
        ])

        self.class_mappings_file = class_mappings_file
        if os.path.exists(self.class_mappings_file):
            self.class_mappings = yaml.safe_load(open(self.class_mappings_file))["mappings"]
        else:
            self.class_mappings = None
            warnings.warn("No class mappings file found. Cannot perform inference without training first.")

    def train(
        self,
        dataset_class: type = LargerFaceRecognitionDataset,
        dataset_path: os.PathLike = os.getenv('DATASET_DIR'),
        save_file: os.PathLike = os.getenv('CLASSIFIER_WEIGHTS_FILE'),
        num_epochs: int = 15,
        learning_rate: float = 0.003,
        batch_size: int = 10,
        transform: Optional[Any] = None,
    ):
        torch.set_float32_matmul_precision('high')
        if transform is not None:
            self.transform = transform
        
        writer = SummaryWriter("logs/")

        self.train_dataset = dataset_class(dataset_path, "train", transform=self.transform)
        self.valid_dataset = dataset_class(dataset_path, "valid", transform=self.transform)
        self.class_mappings = yaml.safe_load(open(self.class_mappings_file))["mappings"]
        
        train_dataloader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)
        validation_dataloader = DataLoader(self.valid_dataset, batch_size=batch_size, shuffle=True)

        loss_function = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(self.classifier_model.parameters(), lr=learning_rate, weight_decay=0.005, momentum=0.9)

        best_valid_accuracy = -1 * float('inf')

        for epoch in range(num_epochs):
            print(f"Epoch {epoch+1}")

            self.classifier_model.train()
            train_losses = []
            samples = 0
            correct_labels = 0

            # Train Loop
            for idx, (image, labels) in enumerate(train_dataloader):
                image = image.to(self.device)
                labels = labels.to(self.device)
                
                optimizer.zero_grad()

                with torch.no_grad():
                    embedding = self.embedding_model(image)

                predictions = self.classifier_model(embedding)
                loss = loss_function(predictions, labels)
                train_losses.append(loss.item())
                loss.backward()
                optimizer.step()

                pred = predictions.argmax(dim=1, keepdim=True)
                label = labels.argmax(dim=1, keepdim=True)
                current_correct = (pred.flatten() == label.flatten()).sum()
                correct_labels += current_correct

                samples += batch_size
                writer.add_scalars("Loss/Sample", {"Train": float(np.mean(train_losses[-batch_size:]))}, samples)
                writer.add_scalars("Accuracy/Sample", {"Train": current_correct * 100.0 / batch_size}, samples)
            
            print(f"Train Loss: {np.mean(train_losses)}")
            print(f"Train Accuracy: {correct_labels} / {samples} = {correct_labels * 100.0 / samples}%")
            writer.add_scalars("Loss/Epoch", {"Train": np.mean(train_losses)}, epoch + 1)
            writer.add_scalars("Accuracy/Epoch", {"Train": correct_labels * 100.0 / samples}, epoch + 1)

            self.classifier_model.eval()
            valid_accuracy = 0.0
            valid_losses = []
            correct_labels, total_labels = 0, 0

            # Validation Loop
            with torch.no_grad():
                for idx, (image, labels) in enumerate(validation_dataloader):
                    image = image.to(self.device)
                    labels = labels.to(self.device)
                    
                    predictions = self.classifier_model(self.embedding_model(image))
                    loss = loss_function(predictions, labels)
                    valid_losses.append(loss.item())

                    pred = predictions.argmax(dim=1, keepdim=True)
                    label = labels.argmax(dim=1, keepdim=True)
                    current_correct = (pred.flatten() == label.flatten()).sum()
                    correct_labels += current_correct
                    total_labels += labels.shape[0]

                    writer.add_scalars("Loss/Sample", {"Valid": float(np.mean(valid_losses[-labels.shape[0]:]))}, total_labels)
                    writer.add_scalars("Accuracy/Sample", {"Valid": current_correct * 100.0 / labels.shape[0]}, total_labels)

            valid_accuracy = correct_labels * 100.0 / total_labels

            print(f"Valid Loss: {np.mean(valid_losses)}")
            print(f"Valid Accuracy: {correct_labels} / {total_labels} = {valid_accuracy}%")
            writer.add_scalars("Loss/Epoch", {"Valid": np.mean(valid_losses)}, epoch + 1)
            writer.add_scalars("Accuracy/Epoch", {"Valid": valid_accuracy}, epoch + 1)

            if valid_accuracy >= best_valid_accuracy:
                best_valid_accuracy = valid_accuracy
                torch.save(self.classifier_model.state_dict(), save_file)
        
        print(f"Best validation accuracy: {best_valid_accuracy}")

    def __call__(self, image: Union[np.ndarray, os.PathLike]):
        self.embedding_model.eval()
        self.classifier_model.eval()
        if isinstance(image, str):
            image_matrix = cv2.imread(image)
        image_matrix = self.transform(image_matrix).to(self.device)
        with torch.no_grad():
            embedding = self.embedding_model(image_matrix).view(1, -1) # Embed and reshape (as a result of batching)
            prediction = self.classifier_model(embedding)
            if self.class_mappings is None:
                raise Exception("Cannot infer if there are no class mappings from training.")
            label = self.class_mappings[prediction.argmax(dim=1, keepdim=True).item()]
            print(label)
            return label

def main(args):

    if args.train:
        fr = FaceRecognizer()

        dataset_class = {
            "larger": LargerFaceRecognitionDataset,
            "custom": CustomFaceRecognitionDataset,
        }[args.dataset_type]

        fr.train(dataset_class=dataset_class, dataset_path=args.dataset_dir)
    else:
        if args.image is not None:
            fr = FaceRecognizer(classifier_weights_file=os.getenv('CLASSIFIER_WEIGHTS_FILE'))
            fr(args.image)
        else:
            warnings.warn("Cannot infer on an unspecified image. Use `--image` to pass image path.")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", dest="train", action="store_true", required=False)
    parser.add_argument("--dataset-dir", type=str, help="Path to dataset directory to train upon", default=os.getenv('DATASET_DIR'), required=False)
    parser.add_argument("--infer", dest="train", action="store_false", required=False)
    parser.add_argument("--image", type=str, help="Path to image to infer upon", default=None, required=False)
    parser.add_argument("--dataset-type", type=str, choices=["custom", "larger"], help="Type of dataset to use for training", default="larger", required=False)
    args = parser.parse_args()
    main(args)
