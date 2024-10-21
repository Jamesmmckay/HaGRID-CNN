import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F 


class AlexNet(nn.Module):
    def __init__(self, num_classes):
        # DEFINE LAYERS, ACTIVATION FUNCTION
        super().__init__() # just run the init of parent class

        self.conv1 = nn.Sequential(nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2),
                                   nn.BatchNorm2d(96),
                                   nn.ReLU(),
                                   nn.MaxPool2d(kernel_size=3, stride=2))

        self.conv2 = nn.Sequential(nn.Conv2d(96, 256, kernel_size=5, padding=2),
                                   nn.BatchNorm2d(256),
                                   nn.ReLU(),
                                   nn.MaxPool2d(kernel_size=3, stride=2))

        self.conv3 = nn.Sequential(nn.Conv2d(256, 384, kernel_size=3, padding=1),
                                   nn.BatchNorm2d(384),
                                   nn.ReLU())

        self.conv4 = nn.Sequential(nn.Conv2d(384, 384, kernel_size=3, padding=1),
                                   nn.BatchNorm2d(384),
                                   nn.ReLU())

        self.conv5 = nn.Sequential(nn.Conv2d(384, 256, kernel_size=3, padding=1),
                                   nn.BatchNorm2d(256),
                                   nn.ReLU(),
                                   nn.MaxPool2d(kernel_size=3, stride=2))

        self.avgpool = torch.nn.AdaptiveAvgPool2d((6, 6)) # pooling-layer

        self.dense1 = nn.Sequential(nn.Dropout(0.5),
                                    nn.Linear(256 * 6 * 6, 4096),
                                    nn.ReLU())
        self.dense2 = nn.Sequential(nn.Dropout(0.5),
                                    nn.Linear(4096, 4096),
                                    nn.ReLU())
        self.dense3 = nn.Sequential(nn.Linear(4096, num_classes))

    def forward(self, xb):
        # Forward pass through the convolutional layers
        out = self.conv1(xb)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)

        # Adaptive pooling to (6, 6)
        out = self.avgpool(out)

        # Flatten the output for the fully connected layers
        out = out.view(out.size(0), -1)  # Use view to flatten; -1 infers the size from the other dimension

        # Forward pass through the dense layers
        out = self.dense1(out)
        out = self.dense2(out)  # Call the second dense layer
        logits = self.dense3(out)  # Call the third dense layer

        return logits