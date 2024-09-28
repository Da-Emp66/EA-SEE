import torch
import torch.nn as nn

class FaceEmbeddingModel(nn.Module):
    def __init__(
        self,
        device: torch.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    ):
        super(FaceEmbeddingModel, self).__init__()
        self.model = nn.Sequential(
            nn.ZeroPad2d(1),
            nn.Conv2d(3, 64, kernel_size=(3,3)),
            nn.ReLU(),
            nn.ZeroPad2d(1),
            nn.Conv2d(64, 64, kernel_size=(3,3)),
            nn.ReLU(),
            nn.MaxPool2d((2,2), stride=(2,2)),

            nn.ZeroPad2d(1),
            nn.Conv2d(64, 128, kernel_size=(3,3)),
            nn.ReLU(),
            nn.ZeroPad2d(1),
            nn.Conv2d(128, 128, kernel_size=(3,3)),
            nn.ReLU(),
            nn.MaxPool2d((2,2), stride=(2,2)),

            nn.ZeroPad2d(1),
            nn.Conv2d(128, 256, kernel_size=(3,3)),
            nn.ReLU(),
            nn.ZeroPad2d(1),
            nn.Conv2d(256, 256, kernel_size=(3,3)),
            nn.ReLU(),
            nn.ZeroPad2d(1),
            nn.Conv2d(256, 256, kernel_size=(3,3)),
            nn.ReLU(),
            nn.MaxPool2d((2,2), stride=(2,2)),

            nn.ZeroPad2d(1),
            nn.Conv2d(256, 512, kernel_size=(3,3)),
            nn.ReLU(),
            nn.ZeroPad2d(1),
            nn.Conv2d(512, 512, kernel_size=(3,3)),
            nn.ReLU(),
            nn.ZeroPad2d(1),
            nn.Conv2d(512, 512, kernel_size=(3,3)),
            nn.ReLU(),
            nn.MaxPool2d((2,2), stride=(2,2)),

            nn.ZeroPad2d(1),
            nn.Conv2d(512, 512, kernel_size=(3,3)),
            nn.ReLU(),
            nn.ZeroPad2d(1),
            nn.Conv2d(512, 512, kernel_size=(3,3)),
            nn.ReLU(),
            nn.ZeroPad2d(1),
            nn.Conv2d(512, 512, kernel_size=(3,3)),
            nn.ReLU(),
            nn.MaxPool2d((2,2), stride=(2,2)),

            nn.Conv2d(512, 4096, kernel_size=(7,7)),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Conv2d(4096, 4096, kernel_size=(1,1)),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Conv2d(4096, 2622, kernel_size=(1,1)),
            nn.Flatten(),
        )
        self.model = self.model.to(device)

    def forward(self, x):
        return self.model(x)

class FaceRecognitionModel(nn.Module):
    def __init__(
        self,
        device: torch.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    ):
        super(FaceRecognitionModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(2622, 100),
            nn.BatchNorm1d(),
            nn.Tanh(),

            nn.Dropout(0.3),

            nn.Linear(100, 10),
            nn.BatchNorm1d(),
            nn.Tanh(),

            nn.Dropout(0.2),

            nn.Linear(10, 6),
            nn.Softmax(),
        )
        self.model = self.model.to(device)

    def forward(self, x):
        return self.model(x)
    