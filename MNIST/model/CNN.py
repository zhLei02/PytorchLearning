import torch
from torch import nn

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 10, kernel_size=5), # (1, 28, 28) -> (10, 24, 24)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2) # (10, 24, 24) -> (10, 12, 12)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(10, 20, kernel_size=5), # (10, 12, 12) -> (20, 8, 8)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2) # (20, 8, 8) -> (20, 4, 4)
        )
        self.fc = nn.Sequential(
            nn.Linear(320, 50), # (bs,320) -> (bs,50)
            nn.ReLU(),
            nn.Linear(50, 10) # (bs,50) -> (bs,10)
        )
    
    def forward(self, x: torch.Tensor):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1) # flatten
        x = self.fc(x)
        return x
