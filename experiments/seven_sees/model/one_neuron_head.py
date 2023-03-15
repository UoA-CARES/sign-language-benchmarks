import torch
import torch.nn as nn

class OneNeuronHead(nn.Module):
    """One neuron head to be used with Seven Sees.
    Args:
        num_modalities (int): Number of modalities being used.
        init_std (float): Std value for Initiation. Default: 0.01.
    """

    def __init__(self, num_modalities, init_std=0.01):
        super(OneNeuronHead, self).__init__()
        self.init_std = init_std
        self.fc_cls = nn.Linear(num_modalities, 1)
        self.init_weights()

    def normal_init(self, module, mean=0, std=1, bias=0):
        nn.init.normal_(module.weight, mean, std)
        if hasattr(module, 'bias') and module.bias is not None:
            nn.init.constant_(module.bias, bias)

    def init_weights(self):
        """Initiate the parameters from scratch."""
        self.normal_init(self.fc_cls, std=self.init_std)

    def forward(self, stream):
        """Forward function for OneNeuronHead made for 7Sees
        
        Args:
            stream (dict): A dictionary containing all the cls_scores for the different
            modalities.
        Returns:
            torch.tensor: The cls_score after going through one neuron."""
        # Concatenate all the modalities
        list_ = []
        
        for modality in stream:
            list_.append(stream[modality])

        x = torch.cat(list_, dim=0)
        
        return self.fc_cls(x.permute(1,0)).permute(1,0)