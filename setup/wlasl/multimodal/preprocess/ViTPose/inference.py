import argparse
import os.path as osp
import os
import torch
from torch import Tensor

from pathlib import Path
import cv2
import numpy as np

import glob
from time import time
from PIL import Image
from torchvision.transforms import transforms

from .models.model import ViTPose
from .utilsvit.visualization import draw_points_and_skeleton, joints_dict
from .utilsvit.dist_util import get_dist_info, init_dist
from .utilsvit.top_down_eval import keypoints_from_heatmaps
'''            "keypoints": {
                0: "nose",
                1: "left_eye",
                2: "right_eye",
                3: "left_ear",
                4: "right_ear",
                5: "left_shoulder",
                6: "right_shoulder",
                7: "left_elbow",
                8: "right_elbow",
                9: "left_wrist",
                10: "right_wrist",
                11: "left_hip",
                12: "right_hip",
                13: "left_knee",
                14: "right_knee",
                15: "left_ankle",
                16: "right_ankle"
'''
@torch.no_grad()
            
class VitPose:
    def __init__(self):
   
        __all__ = ['inference']
        from configs.ViTPose_base_coco_256x192 import model as model_cfg
        from configs.ViTPose_base_coco_256x192 import data_cfg
        self.model_cfg = model_cfg

        
        CUR_DIR = osp.dirname(__file__)
        self.ckpt_path= f"{CUR_DIR}/vitpose-b-multi-coco.pth"
        
        self.img_size = data_cfg['image_size']
                # Prepare model
        self.vit_pose = ViTPose(self.model_cfg)
        self.vit_pose.load_state_dict(torch.load(self.ckpt_path)['state_dict'])
        self.device = 'cuda'
        self.vit_pose.to(self.device)
        
    def inference(self, img, save_result = False):
        
        img_size = self.img_size

        ckpt_path: Path
        device: torch.device 
        

        
        # Prepare input data
        imgs = [img] #glob.glob(img_paths+ os.sep + "*.jpg")
        imgs.sort()
        pointslist = []
        for img_path in imgs:
            
            newh, neww , _ = img_path.shape  

            #check img size not 0
            if( newh == 0 or neww == 0):
                img_path= np.zeros((1,1,3), np.uint8)

            img = Image.fromarray(img_path)#Image.open(img_path)

            

            org_w, org_h = img.size

            print(f">>> Original image size: {org_h} X {org_w} (height X width)")
            print(f">>> Resized image size: {img_size[1]} X {img_size[0]} (height X width)")
            print(f">>> Scale change: {org_h/img_size[1]}, {org_w/img_size[0]}")
            img_tensor = transforms.Compose (
                [transforms.Resize((img_size[1], img_size[0])),
                transforms.ToTensor()]
            )(img).unsqueeze(0).to(self.device)
            
            
            # Feed to model
            tic = time()
            heatmaps = self.vit_pose(img_tensor).detach().cpu().numpy() # N, 17, h/4, w/4
            elapsed_time = time()-tic
            print(f">>> Output size: {heatmaps.shape} ---> {elapsed_time:.4f} sec. elapsed [{elapsed_time**-1: .1f} fps]\n")    
            
            # points = heatmap2coords(heatmaps=heatmaps, original_resolution=(org_h, org_w))
            points, prob = keypoints_from_heatmaps(heatmaps=heatmaps, center=np.array([[org_w//2, org_h//2]]), scale=np.array([[org_w, org_h]]),
                                                unbiased=True, use_udp=True)
            points = np.concatenate([points[:, :, ::-1], prob], axis=2)
            
            # Visualization 
            if save_result:
                for pid, point in enumerate(points):
                    img = np.array(img)[:, :, ::-1] # RGB to BGR for cv2 modules
                    img = draw_points_and_skeleton(img.copy(), point, joints_dict()['coco']['skeleton'], person_index=pid,
                                                points_color_palette='gist_rainbow', skeleton_color_palette='jet',
                                                points_palette_samples=10, confidence_threshold=0.4)
                    save_name = img_path.replace(".jpg", "_result.jpg")
                    cv2.imwrite(save_name, img)
            scaledpoints = []
            points = points.tolist()
            '''
            for person in points:
                personpoints = []
                for point in person:

                    y,x= point[0:2]
                    y,x = y/org_h, x/org_w
                    personpoints.append([y,x,point[2]]) 
                scaledpoints.append(personpoints)
            pointslist.append([img_path.split(os.sep)[-1],scaledpoints])
            '''
        return points[0]#pointslist
        
        

if __name__ == "__main__":
 
    parser = argparse.ArgumentParser()
    parser.add_argument('--image-path', nargs='+', type=str, default='examples/img1.jpg', help='image path(s)')
    args = parser.parse_args()
    Vitpose = VitPose()
    for img_path in args.image_path:
        print(img_path)
        keypoints = Vitpose.inference(img_paths=img_path)