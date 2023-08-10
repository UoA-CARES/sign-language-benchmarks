import torch.nn.functional as F
import torch.nn as nn
import torch


class MultiModalNeck(nn.Module):
    def __init__(self, device='cuda'):
        super(MultiModalNeck, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.device = device

    def forward(self,
                rgb=None,
                depth=None,
                flow=None,
                face=None,
                left_hand=None,
                right_hand=None,
                skeleton=None):

        out = torch.tensor([]).to(self.device)

        if rgb is not None:
            rgb = torch.flatten(self.avg_pool(rgb[-1]), start_dim=1)
            out = torch.concat((out, rgb), dim=1)

        if depth is not None:
            depth = torch.flatten(self.avg_pool(depth[-1]), start_dim=1)
            out = torch.concat((out, depth), dim=1)

        if flow is not None:
            flow = torch.flatten(self.avg_pool(flow[-1]), start_dim=1)
            out = torch.concat((out, flow), dim=1)

        if face is not None:
            face = torch.flatten(self.avg_pool(face[-1]), start_dim=1)
            out = torch.concat((out, face), dim=1)

        if left_hand is not None:
            left_hand = torch.flatten(
                self.avg_pool(left_hand[-1]), start_dim=1)
            out = torch.concat((out, left_hand), dim=1)

        if right_hand is not None:
            right_hand = torch.flatten(
                self.avg_pool(right_hand[-1]), start_dim=1)
            out = torch.concat((out, right_hand), dim=1)

        if skeleton is not None:
            skeleton = torch.flatten(
                self.avg_pool(skeleton[-1]), start_dim=1)
            out = torch.concat((out, skeleton), dim=1)

        return out
