import torch.nn as nn


class FlowAutoencoder(nn.Module):
    """Multimodal Flow model for CSN.
    """

    def __init__(self, rgb_backbone, flow_backbone, neck, head):
        super(FlowAutoencoder, self).__init__()
        self.rgb_backbone = rgb_backbone
        self.flow_backbone = flow_backbone
        self.neck = neck
        self.head = head

    def forward(self, rgb, flow):
        rgb_out = self.rgb_backbone(rgb)
        flow_out = self.flow_backbone(flow)
        neck_out = self.neck(rgb=rgb_out,
                             flow=flow_out
                             )
        cls_score = self.head(neck_out)

        return cls_score
