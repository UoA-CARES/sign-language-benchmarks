import torch.nn as nn


class FlowAutoencoder(nn.Module):
    """Multimodal Flow model for CSN.
    """

    def __init__(self, neck, head, rgb_backbone=None, flow_backbone=None):
        super(FlowAutoencoder, self).__init__()
        self.rgb_backbone = rgb_backbone
        self.flow_backbone = flow_backbone
        self.neck = neck
        self.head = head

    def forward(self, rgb=None, flow=None):
        if rgb is not None:
            rgb = self.rgb_backbone(rgb)
        if flow is not None:
            flow = self.flow_backbone(flow)
        neck_out = self.neck(rgb=rgb,
                             flow=flow
                             )
        cls_score = self.head(neck_out)

        return cls_score
