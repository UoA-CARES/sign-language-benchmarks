import torch.nn.functional as F
import torch.nn as nn
import torch
from .mmcv_csn import ResNet3dCSN
from .pose_encoder import PoseEncoder
from .multimodal_neck import MultiModalNeck
from .cls_head import ClassifierHead

class SevenSeesNet(nn.Module):
    def __init__(self):
        super(SevenSeesNet, self).__init__()

        self.rgb_encoder = ResNet3dCSN(
            pretrained2d=False,
            # pretrained=None,
            pretrained='https://download.openmmlab.com/mmaction/recognition/csn/ircsn_from_scratch_r50_ig65m_20210617-ce545a37.pth',
            depth=50,
            with_pool2=False,
            bottleneck_mode='ir',
            norm_eval=True,
            zero_init_residual=False,
            bn_frozen=True
        )

        self.flow_encoder = ResNet3dCSN(
            pretrained2d=False,
            # pretrained=None,
            pretrained='https://download.openmmlab.com/mmaction/recognition/csn/ircsn_from_scratch_r50_ig65m_20210617-ce545a37.pth',
            depth=50,
            with_pool2=False,
            bottleneck_mode='ir',
            norm_eval=True,
            zero_init_residual=False,
            bn_frozen=True
        )

        
        self.depth_encoder = ResNet3dCSN(
            pretrained2d=False,
            # pretrained=None,
            pretrained='https://download.openmmlab.com/mmaction/recognition/csn/ircsn_from_scratch_r50_ig65m_20210617-ce545a37.pth',
            depth=50,
            with_pool2=False,
            bottleneck_mode='ir',
            norm_eval=True,
            zero_init_residual=False,
            bn_frozen=True
        )

        
        self.lhand_encoder = ResNet3dCSN(
            pretrained2d=False,
            # pretrained=None,
            pretrained='https://download.openmmlab.com/mmaction/recognition/csn/ircsn_from_scratch_r50_ig65m_20210617-ce545a37.pth',
            depth=50,
            with_pool2=False,
            bottleneck_mode='ir',
            norm_eval=True,
            zero_init_residual=False,
            bn_frozen=True
        )

       
        self.rhand_encoder = ResNet3dCSN(
            pretrained2d=False,
            # pretrained=None,
            pretrained='https://download.openmmlab.com/mmaction/recognition/csn/ircsn_from_scratch_r50_ig65m_20210617-ce545a37.pth',
            depth=50,
            with_pool2=False,
            bottleneck_mode='ir',
            norm_eval=True,
            zero_init_residual=False,
            bn_frozen=True
        )

        
        self.face_encoder = ResNet3dCSN(
            pretrained2d=False,
            # pretrained=None,
            pretrained='https://download.openmmlab.com/mmaction/recognition/csn/ircsn_from_scratch_r50_ig65m_20210617-ce545a37.pth',
            depth=50,
            with_pool2=False,
            bottleneck_mode='ir',
            norm_eval=True,
            zero_init_residual=False,
            bn_frozen=True
        )

        # Change this later
        self.pose_encoder = PoseEncoder(1632, 1024)

        self.neck = MultiModalNeck()

        self.head = ClassifierHead(num_classes=400,
                 in_features=13312,
                 dropout_ratio=0.5,
                 init_std=0.01)


    def init_weights(self):
        self.rgb_encoder.init_weights()
        self.depth_encoder.init_weights()
        self.flow_encoder.init_weights()
        self.rhand_encoder.init_weights()
        self.lhand_encoder.init_weights()
        self.face_encoder.init_weights()
        self.pose_encoder.init_weights()
        self.head.init_weights()

    def forward(self,
                rgb=None,
                depth=None,
                flow=None,
                face=None,
                left_hand=None,
                right_hand=None,
                pose=None):
        
        out_rgb = self.rgb_encoder(rgb)[-1]
        out_flow = self.flow_encoder(flow)[-1]
        out_depth = self.depth_encoder(torch.cat((depth, depth, depth), dim=1))[-1]
        out_lhand = self.lhand_encoder(left_hand)[-1]
        out_rhand= self.rhand_encoder(right_hand)[-1]
        out_face = self.face_encoder(face)[-1]
        out_pose = self.pose_encoder(pose)

        out = self.neck(rgb=out_rgb,
          flow=out_flow,
          depth=out_depth,
          face=out_face,
          left_hand=out_lhand,
          right_hand=out_rhand,
          pose=out_pose)

        out = self.head(out)

        return out