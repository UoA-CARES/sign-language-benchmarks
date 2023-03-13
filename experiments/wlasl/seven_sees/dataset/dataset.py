from torch.utils.data import Dataset
import torchvision
import torch

from .pipelines.sampleframes import SampleFrames
from .pipelines.readpose import ReadPose
from .pipelines.normalise import Normalise
import os.path as osp
from PIL import Image
import numpy as np
import cv2

class MultiModalDataset(Dataset):
    """Samples frames using MMAction's SampleFrames and handles multimodal 
    rawframes data.

    Example of a annotation file:
    .. code-block:: txt
        some/directory-1 163 1
        some/directory-2 122 1
        some/directory-3 258 2
        some/directory-4 234 2
        some/directory-5 295 3
        some/directory-6 121 3

    Required keys are "ann_file", "root_dir" and "clip_len".
    Args:
        ann_file (str): Path to annotation file.
        root_dir (str): Root directory of the rawframes.
        clip_len (int): Frames of each sampled output clip.
        frame_interval (int): Temporal interval of adjacent sampled frames.
            Default: 1.
        num_clips (int): Number of clips to be sampled. Default: 1.
        rgb_prefix (str): File format for rgb image files.
        flow_prefix (str): File format for flow image files.
        depth_prefix (str): File format for depth image files.
        test_mode (bool): Store True when building test or validation dataset.
            Default: False.
    """

    def __init__(self,                
                ann_file,
                root_dir,
                clip_len,
                resolution=224,
                input_resolution=256,
                transforms=None,
                frame_interval=1,
                num_clips=1,
                modalities = ('rgb'),
                rgb_prefix =  'img_{:05}.jpg',
                flow_prefix = 'flow_{:05}.jpg',
                depth_prefix = 'depth_{:05}.jpg',
                test_mode=False):

        self.ann_file = ann_file
        self.root_dir = root_dir
        self.rgb_prefix = rgb_prefix
        self.flow_prefix = flow_prefix
        self.input_resolution = input_resolution
        self.depth_prefix = depth_prefix
        self.test_mode = test_mode
        self.transforms = transforms
        self.resolution = resolution
        self.modalities = modalities
        
        self.normalise = Normalise(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        self.flow_normalise = Normalise(mean=[0.9444415, 0.9504853, 0.9530699], std=[
            0.1113386, 0.1044944, 0.1007349])
        
        self.depth_normalise = Normalise(mean=[0.440, 0.440, 0.440], std=[0.226, 0.226, 0.226])
        

        self.video_infos = self.load_annotations()
        self.read_pose = ReadPose()
        self.sample_frames = SampleFrames(clip_len=clip_len,
                                        frame_interval=frame_interval,
                                        num_clips=num_clips,
                                        test_mode=self.test_mode)

        self.img2tensorTransforms = torchvision.transforms.Compose(
                                                [
                                                    torchvision.transforms.ToTensor(),
                                                ]
                                            )

        self.train_transform = torchvision.transforms.Compose([torchvision.transforms.Resize(size=(256)),
                                                               torchvision.transforms.RandomResizedCrop(size=(self.resolution), scale=(0.4, 1.0)),
                                                               torchvision.transforms.RandomHorizontalFlip(p=0.5)
                                                            ]
                                       )
        
        self.resize = torchvision.transforms.Resize(size=(self.resolution, self.resolution))

        self.test_transform = torchvision.transforms.Compose(
            [torchvision.transforms.Resize(size=(256)),
            torchvision.transforms.CenterCrop(size=(self.resolution))])


    def __len__(self):
        return self.numvideos 

    def load_annotations(self):
        """Load annotation file to get video information."""
        video_infos = []
        with open(self.ann_file, 'r') as fin:
            for i, line in enumerate(fin):
                line_split = line.strip().split()

                video_info = dict()
                video_info['video_path'] = osp.join(self.root_dir, line_split[0])
                video_info['start_index'] = 1
                video_info['total_frames'] = int(line_split[1])
                video_info['label'] = int(line_split[2])
                video_infos.append(video_info)
            self.numvideos = i + 1
        return video_infos

    def load_pose(self, video_path):
        """Load pose file under each video to get pose information."""
        pose_frames = dict()
        with open(osp.join(video_path, 'pose.txt'), 'r') as fin:
            for line in fin:
                pose_values, face, lhand, rhand, bodybbox, imgpath = self.read_pose(line)
                pose_frames[int(imgpath[4:9])] = dict(keypoints=pose_values,
                                        face=face,
                                        left_hand=lhand,
                                        right_hand=rhand,
                                        body_bbox=bodybbox,
                                        )
                

        return pose_frames
    
    def load_landmarks(self, results):
        assert 'rgb', 'pose' in self.modalities

        frames = []

        for i, img in enumerate(results['body_bbox']):
            img =  np.array(img)[:, :, ::-1].copy()
            keypoints = results['pose'][i]['keypoints']

            # for j in keypoints:
            #     img = cv2.circle(img, (int(keypoints[j]['x']), int(keypoints[j]['y'])), radius=1, color=(0, 0, 255), thickness=2)

            # Draw lines on the torso and arms
            left_shoulder = (int(keypoints['left_shoulder']['x']), int(keypoints['left_shoulder']['y']))
            right_shoulder = (int(keypoints['right_shoulder']['x']), int(keypoints['right_shoulder']['y']))
            left_hip = (int(keypoints['left_hip']['x']), int(keypoints['left_hip']['y']))
            right_hip = (int(keypoints['right_hip']['x']), int(keypoints['right_hip']['y']))
            left_elbow = (int(keypoints['left_elbow']['x']), int(keypoints['left_elbow']['y']))
            right_elbow = (int(keypoints['right_elbow']['x']), int(keypoints['right_elbow']['y']))
            left_wrist = (int(keypoints['left_wrist']['x']), int(keypoints['left_wrist']['y']))
            right_wrist = (int(keypoints['right_wrist']['x']), int(keypoints['right_wrist']['y']))

            img = cv2.line(img, left_shoulder, right_shoulder, color=(255, 255, 255), thickness=2)
            img = cv2.line(img, left_shoulder, left_hip, color=(255, 255, 255), thickness=2)
            img = cv2.line(img, left_hip, right_hip, color=(255, 255, 255), thickness=2)
            img = cv2.line(img, right_hip, right_shoulder, color=(255, 255, 255), thickness=2)

            img = cv2.line(img, left_shoulder, left_elbow, color=(0, 255, 0), thickness=2)
            img = cv2.line(img, left_elbow, left_wrist, color=(255, 255, 0), thickness=2)
            img = cv2.line(img, right_shoulder, right_elbow, color=(0, 255, 0), thickness=2)
            img = cv2.line(img, right_elbow, right_wrist, color=(255, 255, 0), thickness=2)

            frames.append(self.resize(Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))))

        results['skeleton'] = frames

        return results

    def load_video(self, idx):
        """Load a video at a particular index and return rgb, flow, depth and 
        pose data in a dictionary.
        
        Args: 
            idx (int): The index position in the annotation file
            corresponding to a video.
        Returns:
            results (dict): The dictionary containing all the video data.
        """
        video_info = self.video_infos[idx]
        results = dict()
        results.update(video_info)
        
        self.sample_frames(results)
        frame_indices = results['frame_inds']
        video_path = results['video_path']

        if 'pose' in self.modalities:
            pose_data = self.load_pose(video_path)

        rgb_frames = []
        flow_frames = []
        depth_frames = []
        pose_frames = []

        rgb_frame = None
        flow_frame = None
        depth_frame = None
        pose_frame = None

        cache = dict()

        for frame in frame_indices:
            if frame not in cache:

                if 'rgb' in self.modalities:
                    rgb_frame = Image.open(
                        osp.join(video_path, self.rgb_prefix.format(frame)))

                if 'flow' in self.modalities:
                    flow_frame = Image.open(
                        osp.join(video_path, self.flow_prefix.format(frame)))

                if 'depth' in self.modalities:
                    depth_frame = Image.open(
                        osp.join(video_path, self.depth_prefix.format(frame)))

                if 'pose' in self.modalities:
                    pose_frame = pose_data[frame]

                # Add frames to cache
                cache[frame] = dict(rgb_frame=rgb_frame,
                                depth_frame=depth_frame,
                                flow_frame=flow_frame,
                                pose_frame=pose_frame)
                
                rgb_frames.append(rgb_frame)
                depth_frames.append(depth_frame)
                flow_frames.append(flow_frame)
                pose_frames.append(pose_frame)
                
            else:
                rgb_frames.append(cache[frame]['rgb_frame'])
                depth_frames.append(cache[frame]['depth_frame'])
                flow_frames.append(cache[frame]['flow_frame'])
                pose_frames.append(cache[frame]['pose_frame'])

        if 'rgb' in self.modalities:
            results['rgb'] = rgb_frames

        if 'flow' in self.modalities:
            results['flow'] = flow_frames

        if 'depth' in self.modalities:
            results['depth'] = depth_frames

        if 'pose' in self.modalities:
            results['pose'] = pose_frames
            results = self.crop_part(results, 'body_bbox')

        if 'face' in self.modalities:
            results = self.crop_part(results, 'face')

        if 'left_hand' in self.modalities:
            results = self.crop_part(results, 'left_hand')

        if 'right_hand' in self.modalities:
            results = self.crop_part(results, 'right_hand')
        
        if 'skeleton' in self.modalities:
            results = self.load_landmarks(results)

        return results
        


    def visualise(self, idx=0, key = 'body_bbox'):
        results = self.load_video(idx=idx)
        results = self.transforms(results)  
        for i in range(len(results[key])): 
            img = results[key][i]
            img =  np.array(img)[:, :, ::-1].copy() 
            if(key=='body_bbox'):
                keypoints = results['pose'][i]['keypoints']
                for j in keypoints:
                    img = cv2.circle(img, (int(keypoints[j]['x']), int(keypoints[j]['y'])), radius=1, color=(0, 0, 255), thickness=1)
            cv2.imshow("", img)
            cv2.waitKey(0)

    def to_3dtensor(self, images):
        image_tensors = []
        for img in images:
            image_tensors.append(self.img2tensorTransforms(img).unsqueeze(dim=1))

        tensor = torch.cat(image_tensors, dim = 1)
        return tensor
        
    def pose2tensor(self, pose):
        points =[]
        for posen in pose:
            keypoints = posen['keypoints']           
            for i, point in enumerate(keypoints):
                points.append(keypoints[point]['x'])
                points.append(keypoints[point]['y'])
                points.append(keypoints[point]['confidence'])
        
        tensor = torch.tensor(points)
        return tensor
    
    def crop_part(self, results, part):
        cropped_part = []
        for i in range(len(results['pose'])): 

            # Add padding
            img = results['body_bbox'][i] if part!='body_bbox' else results['rgb'][i]
            w, h = img.size
            img = np.array(img, dtype=np.uint8)

            x0, y0, x1, y1 = [int(value) for value in results['pose'][0][part]]

            if part=='body_bbox':
                try:
                    x0 = 0 if x0<0 else x0
                    x1 = 0 if x0<0 else x1
                    y0 = 0 if x0<0 else y0
                    y1 = 0 if x0<0 else y1
                    img = img[x0:x1, y0:y1]
                except:
                    pass

                img = self.resize(Image.fromarray(img))
                cropped_part.append(img)
                continue
            
            pad_left=0
            pad_right=0
            pad_up=0
            pad_down=0

            if(x0<0):
                pad_left=-x0
            if(x1>w):
                pad_right=x1-w
            if(y0<0):
                pad_up=-y0
            if(y1>h):
                pad_down=y1-h

            x0=pad_left
            x1=w+pad_left
            y0=pad_up
            y1=h+pad_up

            padded = np.zeros((h+pad_up+pad_down, w+pad_left+pad_right, 3), dtype=np.uint8)
            padded[y0:y1, x0:x1] = img

            # Crop the image
            x0, y0, x1, y1 = [int(value) for value in results['pose'][0][part]]

            if x0<0:
                x0=0
                
            if y0<0:
                y0=0

            try:
                if part=='face':
                    # TODO: Fix the coords for head in the preprocess
                    img = Image.fromarray(padded[x0:x1, y0:y1])
                else:
                    # This is the correct crop for all the other modalities
                    img = Image.fromarray(padded[y0:y1, x0:x1])
            except:
                # Erroneous data -> Give full frame
                img = Image.fromarray(padded)

    
            img = self.resize(img)
            cropped_part.append(img)
        
        results[part] = cropped_part

        return results

    def __getitem__(self, idx):
        #['rgb','depth', 'flow', 'pose', 'body_bbox', 'face', 'right_hand','left_hand']
        results = self.load_video(idx=idx)
        modality_list = []
        output = dict()
        
        if 'rgb' in self.modalities:
            rgb = self.to_3dtensor(results['rgb']).squeeze()
        else:
            rgb = torch.FloatTensor(3,32,self.input_resolution,self.input_resolution)

        if 'flow' in self.modalities:
            flow = self.to_3dtensor(results['flow']).squeeze()
        else:
            flow = torch.FloatTensor(3,32,self.input_resolution,self.input_resolution)

        if 'depth' in self.modalities:
            depth = self.to_3dtensor(results['depth'])
            # Since depth is only 1 channel copy it over 3 times
            depth = torch.cat((depth, depth, depth), dim=0)
        else:
            depth = torch.FloatTensor(3,32,self.input_resolution,self.input_resolution)

        if 'face' in self.modalities:
            face = self.to_3dtensor(results['face']).squeeze()
        else:
            face = torch.FloatTensor(3,32,self.resolution,self.resolution)

        if 'left_hand' in self.modalities:
            left_hand = self.to_3dtensor(results['left_hand']).squeeze()
        else:
            left_hand = torch.FloatTensor(3,32,self.resolution,self.resolution)

        if 'right_hand' in self.modalities:
            right_hand = self.to_3dtensor(results['right_hand']).squeeze()
        else:
            right_hand = torch.FloatTensor(3,32,self.resolution,self.resolution)

        if 'skeleton' in self.modalities:
            skeleton = self.to_3dtensor(results['skeleton']).squeeze()
        else:
            skeleton = torch.FloatTensor(3,32,self.resolution,self.resolution)


        modality_list.append(rgb)
        modality_list.append(flow)
        modality_list.append(depth)


        combined_tensor = torch.cat(modality_list, dim=1)

        # Do split based crops
        if self.test_mode:
            combined_tensor = self.test_transform(combined_tensor)
        else:
            combined_tensor = self.train_transform(combined_tensor)
    
        # Don't need crops for face, the hands and skeleton
        combined_tensor = torch.cat([combined_tensor, face], dim=1)
        combined_tensor = torch.cat([combined_tensor, left_hand], dim=1)
        combined_tensor = torch.cat([combined_tensor, right_hand], dim=1)
        combined_tensor = torch.cat([combined_tensor, skeleton], dim=1)

        
    
        if 'rgb' in self.modalities:
            rgb = combined_tensor[:, 0:32, :, :]
            rgb = self.normalise(rgb)
            output['rgb'] = rgb

        if 'flow' in self.modalities:
            flow = combined_tensor[:, 32:64, :, :]
            flow = self.flow_normalise(flow)
            output['flow'] = flow

        if 'depth' in self.modalities:
            depth = combined_tensor[:, 64:96, :, :]
            depth = self.depth_normalise(depth)
            output['depth'] = depth

        if 'face' in self.modalities:
            face = combined_tensor[:, 96:128, :, :]
            face = self.normalise(face)
            output['face'] = face

        if 'left_hand' in self.modalities:
            left_hand = combined_tensor[:, 128:160, :, :]
            left_hand = self.normalise(left_hand)
            output['left_hand'] = left_hand

        if 'right_hand' in self.modalities:
            right_hand = combined_tensor[:, 160:192, :, :]
            right_hand = self.normalise(right_hand)
            output['right_hand'] = right_hand

        if 'skeleton' in self.modalities:
            skeleton = combined_tensor[:, 192:224, :, :]
            skeleton = self.normalise(skeleton)
            output['skeleton'] = skeleton

        label = torch.tensor(results['label'])
        output['label'] = label

        return output



        