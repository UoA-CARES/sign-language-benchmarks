import torch.nn.functional as F
import torch.nn as nn


class PoseEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, init_std=0.01, dropout_ratio=0.1):
        super(PoseEncoder, self).__init__()
        self.init_std = init_std
        self.fc1 = nn.Linear(in_channels, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 1024)
        self.fc4 = nn.Linear(1024, 1024)
        self.fc5 = nn.Linear(1024, 1024)
        self.fc6 = nn.Linear(1024, 1024)
        self.fc7 = nn.Linear(1024, out_channels)
        self.dropout = nn.Dropout(dropout_ratio)

    def normal_init(self, module, mean=0, std=1, bias=0):
        nn.init.normal_(module.weight, mean, std)
        if hasattr(module, 'bias') and module.bias is not None:
            nn.init.constant_(module.bias, bias)

    def init_weights(self):
        """Initiate the parameters from scratch."""
        self.normal_init(self.fc1, std=self.init_std)
        self.normal_init(self.fc2, std=self.init_std)
        self.normal_init(self.fc3, std=self.init_std)
        self.normal_init(self.fc4, std=self.init_std)
        self.normal_init(self.fc5, std=self.init_std)
        self.normal_init(self.fc6, std=self.init_std)
        self.normal_init(self.fc7, std=self.init_std)
        
    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.fc3(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.fc4(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.fc5(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.fc6(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.fc7(x)
        x = F.relu(x)

        return x