import torch.nn as nn


class SimpleHead(nn.Module):
    """Simple head to be used with MultiModalNeck.
    Args:
        num_classes (int): Number of classes to be classified.
        in_channels(int): Number of channels in input feature.
        init_std (float): Std value for Initiation. Default: 0.01.
        dropout_ratio (float): Probability of dropout layer. Default: 0.5.
    """

    def __init__(self,
                 in_channels,
                 num_classes,
                 init_std=0.01,
                 dropout_ratio=0.5
                 ):
        super(SimpleHead, self).__init__()
        self.init_std = init_std
        self.dropout = nn.Dropout(p=dropout_ratio)
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.fc_cls = nn.Linear(self.in_channels, self.num_classes)

    def normal_init(self, module, mean=0, std=1, bias=0):
        nn.init.normal_(module.weight, mean, std)
        if hasattr(module, 'bias') and module.bias is not None:
            nn.init.constant_(module.bias, bias)

    def init_weights(self):
        """Initiate the parameters from scratch."""
        self.normal_init(self.fc_cls, std=self.init_std)

    def forward(self, x):
        """Defines the computation performed at every call.

        Args:
            x (torch.Tensor): The input data.

        Returns:
            torch.Tensor: The classification scores for input samples.
        """
        # [N, in_channels, 1, 1, 1]
        x = self.dropout(x)
        # [N, in_channels, 1, 1, 1]
        x = x.view(x.shape[0], -1)
        # [N, in_channels]
        cls_score = self.fc_cls(x)
        # [N, num_classes]
        return cls_score
