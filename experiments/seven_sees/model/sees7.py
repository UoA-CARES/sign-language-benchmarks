import torch.nn as nn
import torch
class Sees7(nn.Module):
    """Multimodal Model Seven-Sees.
    """

    def __init__(self,
                multistream_backbone=None,
                head=None):
        
        super(Sees7, self).__init__()
        self.multistream_backbone = multistream_backbone

        faceWeight = 0.0
        skeletonWeight = 1
        depthWeigth = 0.0
        flowWeight = 0.0
        rgbWeight = 0
        lefthandWeight = 0.0
        righthandWeight = 0.0
        
        self.modalityWeights = {'rgb':rgbWeight, 'flow': flowWeight, 'depth':depthWeigth, 'skeleton': skeletonWeight, 'face': faceWeight, 'left_hand': lefthandWeight, 'right_hand': righthandWeight}
       
        encoder_layer = nn.TransformerEncoderLayer(d_model=400*7, nhead=8, dim_feedforward = 100)
        self.transformerEncoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        # TODO: Use a head to find the weights for each modality
        self.head = None#self.transformerEncoder
        
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
        

        
        nonZeroModalities  = []
        for modality in self.modalityWeights: #loop modalities
            if( self.modalityWeights[modality] > 0): #check weighting not 0
                weightedScore = stream[modality] * self.modalityWeights[modality] #multiply classification prediction by weight
                nonZeroModalities.append(weightedScore)

        cls_score = nonZeroModalities[0]
        for i in range(1,len(nonZeroModalities)):
            cls_score+= nonZeroModalities[i] 
            #cls_score = torch.concat([cls_score, nonZeroModalities[i]],dim=1)

        if self.head is not None:
            cls_score = self.head(cls_score)
        
        return cls_score