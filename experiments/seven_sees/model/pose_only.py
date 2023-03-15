import torch.nn as nn
from .pose_encoder import PoseEncoder
from .cls_head import ClassifierHead


class PoseOnly(nn.Module):
    def __init__(self):
        super(PoseOnly, self).__init__()

        # Change this later
        self.pose_encoder = PoseEncoder(1632, 2048)

        self.head = ClassifierHead()


        self.head.init_weights()
        self.pose_encoder.init_weights()

    def forward(self, x):
        out_pose = self.pose_encoder(x)      
        return self.head(out_pose)
