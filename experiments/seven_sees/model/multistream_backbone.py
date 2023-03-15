import torch
import torch.nn as nn
from mmcv_model.mmcv_csn import ResNet3dCSN
from model.multimodal_neck import MultiModalNeck
from model.simple_head import SimpleHead
from model.flow_autoencoder import FlowAutoencoder

class MultiStreamBackbone(nn.Module):
    """Multimodal Stream backbone for seven-sees.
    """

    def __init__(self,
                rgb_checkpoint=None,
                flow_checkpoint=None,
                depth_checkpoint=None,
                skeleton_checkpoint=None,
                face_checkpoint=None,
                left_hand_checkpoint=None,
                right_hand_checkpoint=None):
        
        super(MultiStreamBackbone, self).__init__()
        rgb_backbone = ResNet3dCSN(pretrained2d=False,
                                      pretrained=None,
                                      depth=50,
                                      with_pool2=False,
                                      bottleneck_mode='ir',
                                      norm_eval=True,
                                      zero_init_residual=False,
                                      bn_frozen=True
        )
        rgb_neck = MultiModalNeck()
        rgb_head = SimpleHead(num_classes=400,
                              in_channels=2048,
                              dropout_ratio=0.5,
                              init_std=0.01)
        self.rgb_stream = FlowAutoencoder(rgb_backbone=rgb_backbone,
                                     neck=rgb_neck,
                                     head=rgb_head)
        
        if rgb_checkpoint is not None:
            self.rgb_stream.load_state_dict(torch.load(rgb_checkpoint))
    
        flow_backbone = ResNet3dCSN(pretrained2d=False,
                                      pretrained=None,
                                      depth=50,
                                      with_pool2=False,
                                      bottleneck_mode='ir',
                                      norm_eval=True,
                                      zero_init_residual=False,
                                      bn_frozen=True
        )
        flow_neck = MultiModalNeck()
        flow_head = SimpleHead(num_classes=400,
                              in_channels=2048,
                              dropout_ratio=0.5,
                              init_std=0.01)
        self.flow_stream = FlowAutoencoder(flow_backbone=flow_backbone,
                                     neck=flow_neck,
                                     head=flow_head)
        
        if flow_checkpoint is not None:
            self.flow_stream.load_state_dict(torch.load(flow_checkpoint))

        depth_backbone = ResNet3dCSN(pretrained2d=False,
                                      pretrained=None,
                                      depth=50,
                                      with_pool2=False,
                                      bottleneck_mode='ir',
                                      norm_eval=True,
                                      zero_init_residual=False,
                                      bn_frozen=True
        )

        depth_neck = MultiModalNeck()
        depth_head = SimpleHead(num_classes=400,
                              in_channels=2048,
                              dropout_ratio=0.5,
                              init_std=0.01)
        self.depth_stream = FlowAutoencoder(rgb_backbone=depth_backbone,
                                     neck=depth_neck,
                                     head=depth_head)
        
        if depth_checkpoint is not None:
            self.depth_stream.load_state_dict(torch.load(depth_checkpoint))


        skeleton_backbone = ResNet3dCSN(pretrained2d=False,
                                      pretrained=None,
                                      depth=50,
                                      with_pool2=False,
                                      bottleneck_mode='ir',
                                      norm_eval=True,
                                      zero_init_residual=False,
                                      bn_frozen=True
        )
        skeleton_neck = MultiModalNeck()
        skeleton_head = SimpleHead(num_classes=400,
                              in_channels=2048,
                              dropout_ratio=0.5,
                              init_std=0.01)
        self.skeleton_stream = FlowAutoencoder(rgb_backbone=skeleton_backbone,
                                     neck=skeleton_neck,
                                     head=skeleton_head)
        
        if skeleton_checkpoint is not None:
            self.skeleton_stream.load_state_dict(torch.load(skeleton_checkpoint))
            print('Skeleton checkpoint loaded successfully...')
        else:
            print('Skeleton not loaded...')

        face_backbone = ResNet3dCSN(pretrained2d=False,
                                      pretrained=None,
                                      depth=50,
                                      with_pool2=False,
                                      bottleneck_mode='ir',
                                      norm_eval=True,
                                      zero_init_residual=False,
                                      bn_frozen=True
        )
        face_neck = MultiModalNeck()
        face_head = SimpleHead(num_classes=400,
                              in_channels=2048,
                              dropout_ratio=0.5,
                              init_std=0.01)
        self.face_stream = FlowAutoencoder(rgb_backbone=face_backbone,
                                     neck=face_neck,
                                     head=face_head)
        if face_checkpoint is not None:
            self.face_stream.load_state_dict(torch.load(face_checkpoint))

        left_hand_backbone = ResNet3dCSN(pretrained2d=False,
                                      pretrained=None,
                                      depth=50,
                                      with_pool2=False,
                                      bottleneck_mode='ir',
                                      norm_eval=True,
                                      zero_init_residual=False,
                                      bn_frozen=True
        )
        left_hand_neck = MultiModalNeck()
        left_hand_head = SimpleHead(num_classes=400,
                              in_channels=2048,
                              dropout_ratio=0.5,
                              init_std=0.01)
        self.left_hand_stream = FlowAutoencoder(rgb_backbone=left_hand_backbone,
                                     neck=left_hand_neck,
                                     head=left_hand_head)
        if left_hand_checkpoint is not None:
            self.left_hand_stream.load_state_dict(torch.load(left_hand_checkpoint))

        right_hand_backbone = ResNet3dCSN(pretrained2d=False,
                                      pretrained=None,
                                      depth=50,
                                      with_pool2=False,
                                      bottleneck_mode='ir',
                                      norm_eval=True,
                                      zero_init_residual=False,
                                      bn_frozen=True
        )
        right_hand_neck = MultiModalNeck()
        right_hand_head = SimpleHead(num_classes=400,
                              in_channels=2048,
                              dropout_ratio=0.5,
                              init_std=0.01)
        self.right_hand_stream = FlowAutoencoder(rgb_backbone=right_hand_backbone,
                                     neck=right_hand_neck,
                                     head=right_hand_head)
        
        if right_hand_checkpoint is not None:
            self.right_hand_stream.load_state_dict(torch.load(right_hand_checkpoint))


    def forward(self,
                rgb=None,
                flow=None,
                depth=None,
                skeleton=None,
                face=None,
                right_hand=None,
                left_hand=None
                ):
        """Forward method that takes in the different modalities and outputs
        the cls scores of each modalities.
        """
        
        stream = dict()

        if rgb is not None:
            stream['rgb'] = self.rgb_stream(rgb=rgb)
        if flow is not None:
            stream['flow'] = self.flow_stream(flow=flow)
        if depth is not None:
            stream['depth'] = self.depth_stream(rgb=depth)
        if skeleton is not None:
            stream['skeleton'] = self.skeleton_stream(rgb=skeleton)
        if face is not None:
            stream['face'] = self.face_stream(rgb=face)
        if left_hand is not None:
            stream['left_hand'] = self.left_hand_stream(rgb=left_hand)
        if right_hand is not None:
            stream['right_hand'] = self.right_hand_stream(rgb=right_hand)
    
        return stream
