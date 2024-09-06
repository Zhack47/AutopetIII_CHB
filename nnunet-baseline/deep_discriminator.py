import torch

class DeepDiscriminator(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=8, kernel_size=8, stride=2, padding=3, padding_mode="zeros")
        self.conv2 = torch.nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride = 2)
        self.conv3 = torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride = 2)
        self.conv4 = torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2)
        self.conv5 = torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2)
        self.conv6 = torch.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=6)

        self.fc1 = torch.nn.Linear(in_features=256, out_features=128)
        self.fc2 = torch.nn.Linear(in_features=128, out_features=64)
        self.fc3 = torch.nn.Linear(in_features=64, out_features=16)
        self.fc4 = torch.nn.Linear(in_features=16, out_features=4)
        self.fc5 = torch.nn.Linear(in_features=4, out_features=1)
    
    def __call__(self, x):
        return self.forward(x)

    def forward(self,x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = torch.relu(self.conv4(x))
        x = torch.relu(self.conv5(x))
        x = torch.relu(self.conv6(x))

        x = torch.flatten(x, start_dim=1)

        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = self.fc5(x)

        x = torch.sigmoid(x)
        return torch.squeeze(x)