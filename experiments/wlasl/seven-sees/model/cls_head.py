import torch.nn as nn


class ClassifierHead(nn.Module):
    """Classification head for CSN.
    Args:
        num_classes (int): Number of classes to be classified.
        in_features(int): Number of channels in input feature.
        init_std (float): Std value for Initiation. Default: 0.01.
        dropout_ratio (float): Probability of dropout layer. Default: 0.5.
    """

    def __init__(self, in_features=2048, num_classes=400, init_std=0.01, dropout_ratio=0.1):
        super(ClassifierHead, self).__init__()
        self.init_std = init_std
        self.dropout = nn.Dropout(p=dropout_ratio)
        self.fc1 = nn.Linear(in_features, 5000)
        self.fc2 = nn.Linear(5000, 1000)
        self.fc3 = nn.Linear(1000, num_classes)

    def normal_init(self, module, mean=0, std=1, bias=0):
        nn.init.normal_(module.weight, mean, std)
        if hasattr(module, 'bias') and module.bias is not None:
            nn.init.constant_(module.bias, bias)

    def init_weights(self):
        """Initiate the parameters from scratch."""
        self.normal_init(self.fc1, std=self.init_std)
        self.normal_init(self.fc2, std=self.init_std)
        self.normal_init(self.fc3, std=self.init_std)

    def forward(self, x):
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x = self.fc3(x)

        return x
