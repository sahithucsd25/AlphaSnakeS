import torch
import torch.nn as nn


class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels=2, out_channels=8, kernel_size=16, stride=8, padding=1
        )

        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=8, stride=4)

        # Corrected the calculation of the size of the flattened layer
        self.fc1_input_size = 16 * 6 * 6

        self.fc1 = nn.Linear(self.fc1_input_size, 128)

        self.fc2 = nn.Linear(128, 4)  # Adjust as needed for multiple snakes or actions

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = x.view(
            -1, self.fc1_input_size
        )  # Flatten the tensor for the fully connected layer
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
