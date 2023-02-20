import os
import torch
import numpy as np
import cv2



os.chdir('../../../..')

model_type = "DPT_Large"     # MiDaS v3 - Large     (highest accuracy, slowest inference speed)
# model_type = "DPT_Hybrid"   # MiDaS v3 - Hybrid    (medium accuracy, medium inference speed)
# model_type = "MiDaS_small"  # MiDaS v2.1 - Small   (lowest accuracy, highest inference speed)

midas = torch.hub.load("intel-isl/MiDaS", model_type)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
midas.to(device)
midas.eval()

midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
    transform = midas_transforms.dpt_transform
else:
    transform = midas_transforms.small_transform


def get_depth(img):
    '''Returns the depth using the MiDaS model.
    Args:
        img (torch.tensor): Input image in CHW format.
    Returns:
        prediction (torch.tensor) Depth image in 2HW format.
    '''
    with torch.no_grad():
        prediction = midas(img)
   
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),            
            size=tuple((256,256)),
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    return prediction

def write_depth(depth, frame_number, out_path):
    # Format is depth_00001.jpg
    cv2.imwrite(os.path.join(out_path, f'depth_{frame_number:05d}.jpg'), depth.cpu().numpy())


def process_video(video_dir):
    '''Processes all the frames in a video directory.
    Args:
        video_dir (str): The directory of the video.
    '''
    frames = [img for img in os.listdir(video_dir) if img[:3]=='img']
    for i, frame in enumerate(frames):
        img = cv2.imread(os.path.join(video_dir,frame))
        img = np.array(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        img_transform = transform(img).to(device)
        # Multiply by 8 so values are retained even when normalised
        depth = get_depth(img_transform)*8
        write_depth(depth, i+1, video_dir)


def process_subset(subset_dir):
    '''Processes all the frames for video directories under a subset directory.
    Args:
        subset_dir (str): The root directory for the video folders.
    '''
    for video in os.listdir(subset_dir):
        video_dir = os.path.join(subset_dir, video)
        process_video(video_dir)
        print(video_dir)


if __name__ == '__main__':
    # Extract depth for wlasl
    data = 'data/wlasl/rawframes/'
    # subsets = ['train', 'test', 'val']
    subsets = ['train', 'val', 'test']


    for subset in subsets:
        subset_dir = os.path.join(data, subset)
        process_subset(subset_dir)
