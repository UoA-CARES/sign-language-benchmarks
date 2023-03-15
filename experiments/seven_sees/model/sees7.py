import torch.nn as nn

class Sees7(nn.Module):
    """Multimodal Model Seven-Sees.
    """

    def __init__(self,
                multistream_backbone=None,
                head=None):
        
        super(Sees7, self).__init__()
        self.multistream_backbone = multistream_backbone
        self.rgbWeight = 0
        self.flowWeight = 0
        self.depthWeight = 0
        self.poseWeight = 0
        self.faceWeight = 0
        self.leftHandWeight = 1
        self.rightHandWeight = 1
        # TODO: Use a head to find the weights for each modality
        self.head = head
        
    def forward(self,
                rgb=None,
                flow=None,
                depth=None,
                skeleton=None,
                face=None,
                right_hand=None,
                left_hand=None
                ):
        
        # Get a dict containing the cls_scores from the streams
        stream = self.multistream_backbone(rgb=rgb,
                                           flow=flow,
                                           depth=depth,
                                           skeleton=skeleton,
                                           face=face,
                                           right_hand=right_hand,
                                           left_hand=left_hand)
        

        if self.head is None:
            cls_score = self.rgbWeight*(1/7)*stream['rgb'] + self.flowWeight*(1/7)*stream['flow'] + self.depthWeight * (1/7)*stream['depth']
            + self.poseWeight * (1/7)*stream['skeleton'] + self.faceWeight* (1/7)*stream['face'] + self.leftHandWeight * (1/7)*stream['left_hand']
            + self.rightHandWeight * (1/7)*stream['right_hand']
        else:
            cls_score = self.head(stream)

        return cls_score