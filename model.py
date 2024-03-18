import torch
import torch.nn as nn


class DQN(nn.Module):
    def __init__(self, width, device, stack_size=2):
        super(DQN, self).__init__()

        if width == 24:# input size 24x24 (240x240)
            self.conv1 = nn.Conv2d(in_channels=stack_size, out_channels=8, kernel_size=16, stride=8, padding=1)
            self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=8, stride=4)
        elif width == 15: # input size 15x15 (150x150)
            self.conv1 = nn.Conv2d(in_channels=stack_size, out_channels=2, kernel_size=10, stride=10) # 15x15
            self.conv2 = nn.Conv2d(in_channels=2, out_channels=8, kernel_size=3, stride=2, padding=1) # 8x8
            self.conv3 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=2, stride=2) # 4x4
        elif width == 10: # input size 10x10 (100x100)
            self.conv1 = nn.Conv2d(in_channels=stack_size, out_channels=2, kernel_size=10, stride=10) # 10x10
            self.conv2 = nn.Conv2d(in_channels=2, out_channels=8, kernel_size=3, stride=1) # 8x8
            self.conv3 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=2, stride=2) # 4x4

        # Corrected the calculation of the size of the flattened layer
        # self.fc1_input_size = 16 * 6 * 6 # for input size 24x24
        self.fc1_input_size = 16 * 4 * 4

        self.fc1 = nn.Linear(self.fc1_input_size, 128)

        self.fc2 = nn.Linear(128, 4)  # Adjust as needed for multiple snakes or actions

        self.relu = nn.ReLU(inplace=True)

        self.device = device
        self.to(self.device)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = x.view(
            -1, self.fc1_input_size
        )  # Flatten the tensor for the fully connected layer
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

