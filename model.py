import torch
import torch.nn as nn


class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=2, out_channels=8, kernel_size=4, stride=2, padding=1)
        
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=2, stride=2)
        
        # Calculate the size of the flattened layer
        # Starting size: 24x24, after first conv: ((24 - 4 + 2*1)/2 + 1) = 12x12, after second conv: ((12 - 2)/2 + 1) = 6x6
        self.fc1_input_size = 16 * 6 * 6
        
        self.fc1 = nn.Linear(self.fc1_input_size, 128)
        
        self.fc2 = nn.Linear(128, 4) # needs to be changed for multiple snakes?

        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = x.view(-1, self.fc1_input_size)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)

        return x



# example use: model = CustomizableDQN(input_size=10, output_size=4, hidden_layers=[128, 256, 128], activation_function='ReLU') 