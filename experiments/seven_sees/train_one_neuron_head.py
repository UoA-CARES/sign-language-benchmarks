import os
import torch
import wandb
import torch.nn as nn
import numpy as np

from model.multistream_backbone import MultiStreamBackbone
from model.sees7 import Sees7
from model.one_neuron_head import OneNeuronHead
from dataset.dataset import MultiModalDataset
from mmcv_model.scheduler import GradualWarmupScheduler

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

def train_one_epoch(epoch_index, interval=5):
    """Run one epoch for training.
    Args:
        epoch_index (int): Current epoch.
        interval (int): Frequency at which to print logs.
    Returns:
        last_loss (float): Loss value for the last batch.
    """
    running_loss = 0.
    last_loss = 0.

    for i, results in enumerate(train_loader):
        rgb = results['rgb']
        flow = results['flow']
        depth = results['depth']
        face = results['face']
        skeleton = results['skeleton']
        right_hand = results['right_hand']
        left_hand = results['left_hand']
        targets = results['label']

        targets = targets.reshape(-1, )

        rgb, flow, targets = rgb.to(device), flow.to(device), targets.to(device)
        depth, face, skeleton = depth.to(device), face.to(device), skeleton.to(device)
        left_hand, right_hand = left_hand.to(device), right_hand.to(device)

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs = model(rgb=rgb,
                        flow=flow,
                        depth=depth,
                        left_hand=left_hand,
                        right_hand=right_hand,
                        face=face,
                        skeleton=skeleton)

        # Compute the loss and its gradients
        loss = loss_fn(outputs, targets)
        loss.backward()

        # Gradient Clipping
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=40, norm_type=2.0)

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        if i % interval == interval-1:
            last_loss = running_loss / interval  # loss per batch
            print(
                f'Epoch [{epoch_index}][{i+1}/{len(train_loader)}], lr: {scheduler.get_last_lr()[0]:.5e}, loss: {last_loss:.5}')
            running_loss = 0.

    return last_loss, scheduler.get_last_lr()[0]

def validate():
    """Run one epoch for validation.
    Returns:
        avg_vloss (float): Validation loss value for the last batch.
        top1_acc (float): Top-1 accuracy in decimal.
        top5_acc (float): Top-5 accuracy in decimal.
    """
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

    wandb.init(entity="cares", project="jack-slr",
            group="average", name="late-one-fc")
    
    
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
        if("fc" not in name):
            para.requires_grad = False

    model = Sees7(multistream_backbone=multistream)#,
                  #head=OneNeuronHead(num_modalities=7))
    model.to(device)

    
    # Build the dataloaders
    os.chdir('../../')
    work_dir = 'work_dirs/sees7/'
    os.makedirs(work_dir, exist_ok=True)
    batch_size = 1

    train_dataset = MultiModalDataset(ann_file='data/wlasl/train_annotations.txt',
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
                                    test_mode=False,
                                    frame_interval=1,
                                    input_resolution=256,
                                    num_clips=1
                                    )
    
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
    
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                            batch_size=batch_size,
                                            shuffle=True,
                                            num_workers=4,
                                            pin_memory=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                            batch_size=1,
                                            shuffle=True,
                                            num_workers=4,
                                            pin_memory=True)
    
    epochs = 400

    # Specify optimizer
    optimizer = torch.optim.SGD(
        model.parameters(), lr=0.000125, momentum=0.9, weight_decay=0.00001)
    
    # Specify learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=120, gamma=0.1)

    scheduler_steplr = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[34, 84], gamma=0.1)
    scheduler = GradualWarmupScheduler(
        optimizer, multiplier=1, total_epoch=16, after_scheduler=scheduler_steplr)

    # Specify Loss
    loss_fn = nn.CrossEntropyLoss()
    

    for epoch in range(epochs):
        # Turn on gradient tracking and do a forward pass
        model.train(True)

        # Freeze the backbones
        for name, para in model.multistream_backbone.named_parameters():
            if("fc" not in name):
                para.requires_grad = False

        avg_loss, learning_rate = train_one_epoch(epoch+1)

        # Turn off  gradients for reporting
        model.train(False)

        top1_acc, top5_acc = validate()

        print(
            f'top1_acc: {top1_acc:.4}, top5_acc: {top5_acc:.4}, train_loss: {avg_loss:.5}')

        # Track best performance, and save the model's state
        model_path = work_dir + f'epoch_{epoch+1}.pth'
        print(f'Saving checkpoint at {epoch+1} epochs...')
        torch.save(model.state_dict(), model_path)

        # Adjust learning rate
        scheduler.step()
    
        # Track wandb
        wandb.log({'train/loss': avg_loss,
                'train/learning_rate': learning_rate,
                'val/top1_accuracy': top1_acc,
                'val/top5_accuracy': top5_acc})