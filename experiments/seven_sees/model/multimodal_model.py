import torch.nn as nn


class MultiModalModel(nn.Module):
    """Multimodal Flow model for CSN.
    """

    def __init__(self,
                neck,
                head,
                rgb_backbone=None,
                flow_backbone=None,
                depth_backbone=None,
                skeleton_backbone=None,
                face_backbone=None,
                left_hand_backbone=None,
                right_hand_backbone=None):
        
        super(MultiModalModel, self).__init__()
        self.rgb_backbone = rgb_backbone
        self.flow_backbone = flow_backbone
        self.depth_backbone = depth_backbone
        self.skeleton_backbone = skeleton_backbone
        self.face_backbone = face_backbone
        self.left_hand_backbone = left_hand_backbone
        self.right_hand_backbone = right_hand_backbone
        self.neck = neck
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
        if rgb is not None:
            rgb = self.rgb_backbone(rgb)
        if flow is not None:
            flow = self.flow_backbone(flow)
        if depth is not None:
            depth = self.depth_backbone(depth)
        if skeleton is not None:
            skeleton = self.skeleton_backbone(skeleton)
        if face is not None:
            face = self.face_backbone(face)
        if left_hand is not None:
            left_hand = self.left_hand_backbone(left_hand)
        if right_hand is not None:
            right_hand = self.right_hand_backbone(right_hand)
        
        neck_out = self.neck(rgb=rgb,
                             flow=flow,
                             depth=depth,
                             skeleton=skeleton,
                             face=face,
                             left_hand=left_hand,
                             right_hand=right_hand
                             )
        cls_score = self.head(neck_out)

        return cls_score
