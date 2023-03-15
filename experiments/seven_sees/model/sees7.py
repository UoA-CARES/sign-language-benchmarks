import torch.nn as nn

class Sees7(nn.Module):
    """Multimodal Model Seven-Sees.
    """

    def __init__(self,
                multistream_backbone=None,
                head=None):
        
        super(Sees7, self).__init__()
        self.multistream_backbone = multistream_backbone

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
            cls_score = (1/7)*stream['rgb'] + (1/7)*stream['flow'] + (1/7)*stream['depth']
            + (1/7)*stream['skeleton'] + (1/7)*stream['face'] + (1/7)*stream['left_hand']
            + (1/7)*stream['right_hand']
        else:
            cls_score = self.head(stream)

        return cls_score