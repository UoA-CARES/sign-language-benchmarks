import torch
import torchvision.transforms.functional as F
import os
import cv2

from torchvision.models.optical_flow import Raft_Large_Weights
from torchvision.models.optical_flow import raft_large
from torchvision.utils import flow_to_image

os.chdir('../../../..')

# Load the weights for RAFT model
weights = Raft_Large_Weights.DEFAULT
transforms = weights.transforms()

# Run device agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the raft large model
model = raft_large(weights=Raft_Large_Weights.DEFAULT, progress=False).to(device)
model = model.eval()


def load_images(video_path, start_frame, end_frame):
    '''Loads the start image and the end image from the video path where the
    images are in img_00001.jpg format.
    
    Args:
        video_path (str): The path to the folder containing the frames.
        start_frame (int): The frame number for the first image to be loaded i.e.
            1 is the value for start_frame to load img_00001.jpg.
        end_frame(int): The frame number for the last second to be loaded i.e.
            4 is the value for end_frame to load img_00004.jpg.
            
    Returns:
        img1_batch (torch.tensor): The first image loaded as NCHW.
        img2_batch (torch.tensor): The second image loaded as NCHW.
    '''
    image1_path = os.path.join(video_path, f'img_{start_frame:05d}.jpg')
    image2_path = os.path.join(video_path, f'img_{end_frame:05d}.jpg')

    img1 = cv2.cvtColor(cv2.imread(image1_path), cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(cv2.imread(image2_path), cv2.COLOR_BGR2RGB)

    img1_batch = torch.tensor(img1).unsqueeze(dim=0).permute(0,3,1,2)
    img2_batch = torch.tensor(img2).unsqueeze(dim=0).permute(0,3,1,2)
                               
    return img1_batch, img2_batch

def preprocess(img1_batch, img2_batch):
    '''Resize and apply transforms for the RAFT Large model.

    Args: 
        img1_batch (torch.tensor): The first image loaded as NCHW.
        img2_batch (torch.tensor): The second image loaded as NCHW.
        
    Returns:
        img1_batch (torch.tensor): The first image transformed.
        img2_batch (torch.tensor): The second image transformed.
    '''
    img1_batch = F.resize(img1_batch, size=[256, 256])
    img2_batch = F.resize(img2_batch, size=[256, 256])
    return transforms(img1_batch, img2_batch)

def get_flow(img1_batch, img2_batch):
    with torch.inference_mode():
        return model(img1_batch.to(device), img2_batch.to(device))[-1]

def write_img(flow_img, frame_number, out_path):
    cv2.imwrite(os.path.join(out_path, f'flow_{frame_number:05d}.jpg'),
                flow_img.squeeze().permute(1,2,0).cpu().numpy())

def process_video(video_path, gap=2):
    '''Write flow images for a video.
    
    Args:
        video_path (str): The path to the folder containing all the frames for a video.
        gap (int): The number of frames to skip for finding optical flow. Default: 2
    '''
    start_frame = 1
    frame_number = 1
    end_frame = gap
    n_frames = len([img for img in os.listdir(video_path) if img[:3]=='img'])

    while end_frame <= n_frames:
        img1, img2 = load_images(video_path, start_frame, end_frame)
        img1, img2 = preprocess(img1, img2)
        flow = get_flow(img1, img2)
        flow_img = flow_to_image(flow)
        write_img(flow_img, frame_number, video_path)
        
        frame_number += 1
        start_frame = end_frame
        end_frame += gap-1

    # Write last flow frame again to match the number of frames
    write_img(flow_img, frame_number, video_path)

def process_subset(subset_path):
    '''Write flow images for all the video directories under a root path.
    
    Args:
        subset_path (str): The root path containing all the video folders where each folder
        contains all the frames for that video.
    '''
    for video in os.listdir(subset_path):
        video_path = os.path.join(subset_path, video)
        process_video(video_path)
        print(video_path)

if __name__ == '__main__':
    # Extract flow for wlasl
    data = 'data/wlasl/rawframes/'
    # subsets = ['train', 'test', 'val']
    subsets = ['val', 'train', 'test']
    for subset in subsets:
        subset_dir = os.path.join(data, subset)
        process_subset(subset_dir)
