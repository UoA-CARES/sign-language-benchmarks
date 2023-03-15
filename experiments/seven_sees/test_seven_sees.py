import os
import torch
import torch.nn as nn
import numpy as np

from model.multistream_backbone import MultiStreamBackbone
from model.sees7 import Sees7
from dataset.dataset import MultiModalDataset

def top_k_accuracy(scores, labels, topk=(1, )):
    """Calculate top k accuracy score.
    Args:
        scores (list[np.ndarray]): Prediction scores for each class.
        labels (list[int]): Ground truth labels.
        topk (tuple[int]): K value for top_k_accuracy. Default: (1, ).
    Returns:
        list[float]: Top k accuracy score for each k.
    """
    res = np.zeros(len(topk))
    labels = np.array(labels)[:, np.newaxis]
    for i, k in enumerate(topk):
        max_k_preds = np.argsort(scores, axis=1)[:, -k:][:, ::-1]
        match_array = np.logical_or.reduce(max_k_preds == labels, axis=1)
        topk_acc_score = match_array.sum() / match_array.shape[0]
        res[i] = topk_acc_score

    return res

def validate():
    """Run one epoch for validation.
    Returns:
        avg_vloss (float): Validation loss value for the last batch.
        top1_acc (float): Top-1 accuracy in decimal.
        top5_acc (float): Top-5 accuracy in decimal.
    """
    running_vloss = 0.0
    running_vacc = np.zeros(2)

    print('Evaluating top_k_accuracy...')

    model.eval()
    with torch.inference_mode():
        for i, results in enumerate(test_loader):
            rgb = results['rgb']
            flow = results['flow']
            depth = results['depth']
            face = results['face']
            skeleton = results['skeleton']
            right_hand = results['right_hand']
            left_hand = results['left_hand']
            vtargets = results['label']

            vtargets = vtargets.reshape(-1, )

            rgb, flow, vtargets = rgb.to(device), flow.to(device), vtargets.to(device)
            depth, face, skeleton = depth.to(device), face.to(device), skeleton.to(device)
            left_hand, right_hand = left_hand.to(device), right_hand.to(device)

            voutputs = model(rgb=rgb,
                             flow=flow,
                            depth=depth,
                            left_hand=left_hand,
                            right_hand=right_hand,
                            face=face,
                            skeleton=skeleton)


            running_vacc += top_k_accuracy(voutputs.detach().cpu().numpy(),
                                           vtargets.detach().cpu().numpy(), topk=(1, 5))

    acc = running_vacc/len(test_loader)
    top1_acc = acc[0].item()
    top5_acc = acc[1].item()

    return (top1_acc, top5_acc)

if __name__=='__main__':

    device = 'cuda'
    
    # Building the model
    multistream = MultiStreamBackbone(rgb_checkpoint='./rgb.pth',
                                    flow_checkpoint='./flow.pth',
                                    depth_checkpoint='./depth.pth',
                                    skeleton_checkpoint='./skeleton.pth',
                                    face_checkpoint='./face.pth',
                                    left_hand_checkpoint='./left_hand.pth',
                                    right_hand_checkpoint='./right_hand.pth'
                                    )
    

    # Freeze the backbones
    for name, para in multistream.named_parameters():
        para.requires_grad = False

    model = Sees7(multistream_backbone=multistream)
    model.to(device)

    os.chdir('../../')
    # Build the dataloaders
    test_dataset = MultiModalDataset(ann_file='data/wlasl/test_annotations.txt',
                                    root_dir='data/wlasl/rawframes',
                                    clip_len=32,
                                    resolution=224,
                                    modalities=('rgb',
                                                'flow',
                                                'depth',
                                                'pose',
                                                'skeleton',
                                                'face',
                                                'left_hand',
                                                'right_hand'
                                                ),
                                    test_mode=True,
                                    frame_interval=1,
                                    input_resolution=256,
                                    num_clips=1
                                    )

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                            batch_size=1,
                                            shuffle=True,
                                            num_workers=4,
                                            pin_memory=True)
    
    # Evaluate
    top1_acc, top5_acc = validate()
    print(f'top1_acc: {top1_acc:.4}, top5_acc: {top5_acc:.4}')

