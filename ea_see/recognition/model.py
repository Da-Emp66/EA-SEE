import torch.nn as nn

class FaceEmbeddingModel(nn.Module):
    def __init__(self):
        super(FaceEmbeddingModel, self).__init__()
        # self.model = nn.Sequential(
        #     nn.ZeroPad2d(1),
        #     nn.Conv2d(3, 64, kernel_size=(3,3)),
        #     nn.ReLU(),
        #     nn.ZeroPad2d(1),
        #     nn.Conv2d(64, 64, kernel_size=(3,3)),
        #     nn.ReLU(),
        #     nn.MaxPool2d((2,2), stride=(2,2)),

        #     nn.ZeroPad2d(1),
        #     nn.Conv2d(64, 128, kernel_size=(3,3)),
        #     nn.ReLU(),
        #     nn.ZeroPad2d(1),
        #     nn.Conv2d(128, 128, kernel_size=(3,3)),
        #     nn.ReLU(),
        #     nn.MaxPool2d((2,2), stride=(2,2)),

        #     nn.ZeroPad2d(1),
        #     nn.Conv2d(128, 256, kernel_size=(3,3)),
        #     nn.ReLU(),
        #     nn.ZeroPad2d(1),
        #     nn.Conv2d(256, 256, kernel_size=(3,3)),
        #     nn.ReLU(),
        #     nn.ZeroPad2d(1),
        #     nn.Conv2d(256, 256, kernel_size=(3,3)),
        #     nn.ReLU(),
        #     nn.MaxPool2d((2,2), stride=(2,2)),

        #     nn.ZeroPad2d(1),
        #     nn.Conv2d(256, 512, kernel_size=(3,3)),
        #     nn.ReLU(),
        #     nn.ZeroPad2d(1),
        #     nn.Conv2d(512, 512, kernel_size=(3,3)),
        #     nn.ReLU(),
        #     nn.ZeroPad2d(1),
        #     nn.Conv2d(512, 512, kernel_size=(3,3)),
        #     nn.ReLU(),
        #     nn.MaxPool2d((2,2), stride=(2,2)),

        #     nn.ZeroPad2d(1),
        #     nn.Conv2d(512, 512, kernel_size=(3,3)),
        #     nn.ReLU(),
        #     nn.ZeroPad2d(1),
        #     nn.Conv2d(512, 512, kernel_size=(3,3)),
        #     nn.ReLU(),
        #     nn.ZeroPad2d(1),
        #     nn.Conv2d(512, 512, kernel_size=(3,3)),
        #     nn.ReLU(),
        #     nn.MaxPool2d((2,2), stride=(2,2)),

        #     nn.Conv2d(512, 4096, kernel_size=(7,7)),
        #     nn.ReLU(),
        #     nn.Dropout(0.5),
        #     nn.Conv2d(4096, 4096, kernel_size=(1,1)),
        #     nn.ReLU(),
        #     nn.Dropout(0.5),
        #     nn.Conv2d(4096, 2622, kernel_size=(1,1)),
        #     nn.Flatten(),
        # )

        self.model = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
        )

    def forward(self, x):
        return self.model(x)
    

class FaceRecognitionModel(nn.Module):
    def __init__(self, num_classes: int):
        super(FaceRecognitionModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(25088, 4622),
            nn.BatchNorm1d(4622),
            nn.Tanh(),

            nn.Dropout(0.3),

            nn.Linear(4622, 4622),
            nn.BatchNorm1d(4622),
            nn.Tanh(),

            nn.Dropout(0.2),

            nn.Linear(4622, num_classes),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        return self.model(x)
