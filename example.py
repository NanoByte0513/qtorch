import torch
import torch.nn as nn
import numpy as np
import qtorch
# from torch.onnx.symbolic_registry import regit


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.layer0 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer1 = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc = nn.Linear(784, 10)

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = x.view(-1, 784)
        x = self.fc(x)
        return x

net = Net()
input_x = torch.randn(1, 1, 28, 28)
output = net(input_x)

with torch.no_grad():
    torch.onnx.export(net, input_x, "net.onnx", opset_version=11, input_names=['input'], output_names=['output'])