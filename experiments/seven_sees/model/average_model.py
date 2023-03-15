import torch.nn as nn


class MultiModalModel(nn.Module):
    """Multimodal Flow model for CSN.
    """

    def __init__(self,
                rgb_stream=None,
                flow_stream=None,
                depth_stream=None,
                skeleton_stream=None,
                face_stream=None,
                left_hand_stream=None,
                right_hand_stream=None):
        
        super(MultiModalModel, self).__init__()
        self.rgb_stream = rgb_stream
        self.flow_stream = flow_stream
        self.depth_stream = depth_stream
        self.skeleton_stream = skeleton_stream
        self.face_stream = face_stream
        self.left_hand_stream = left_hand_stream
        self.right_hand_stream = right_hand_stream
        
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
            rgb = self.rgb_stream(rgb=rgb)
        if flow is not None:
            flow = self.flow_stream(flow=flow)
        if depth is not None:
            depth = self.depth_stream(depth)
        if skeleton is not None:
            skeleton = self.skeleton_stream(skeleton)
        if face is not None:
            face = self.face_stream(face)
        if left_hand is not None:
            left_hand = self.left_hand_stream(left_hand)
        if right_hand is not None:
            right_hand = self.right_hand_stream(right_hand)
        

        cls_score = 0.7*rgb+0.3*flow

        return cls_score
