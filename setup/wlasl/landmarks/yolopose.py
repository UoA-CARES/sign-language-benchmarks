import torch
from torchvision import transforms

from yolov7pose.utils.datasets import letterbox
from yolov7pose.utils.general import non_max_suppression_kpt
from yolov7pose.utils.plots import output_to_keypoint, plot_skeleton_kpts

import matplotlib.pyplot as plt
import cv2
import numpy as np
import sys

class yolopose:
    def __init__(self):
        sys.path.insert(0, './yolov7pose')
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = torch.load('./yolov7pose/yolov7-w6-pose.pt', map_location=self.device)['model']
        # Put in inference mode
        self.model.float().eval()

        if torch.cuda.is_available():
            # half() turns predictions into float16 tensors
            # which significantly lowers inference time
            self.model.half().to(self.device)


    def run_inference(self,image):
        #image = cv2.imread(url) # shape: (480, 640, 3)
        # Resize and pad image
        image = letterbox(image, 960, stride=64, auto=True)[0] # shape: (768, 960, 3)
        # Apply transforms
        image = transforms.ToTensor()(image) # torch.Size([3, 768, 960])
        # Turn image into batch
        image = image.unsqueeze(0) # torch.Size([1, 3, 768, 960])
        output, _ = self.model(image.half().to(self.device)) # torch.Size([1, 45900, 57])
        return output, image

    def visualize_output(self, output, image):
        output = non_max_suppression_kpt(output,
                                         0.25, # Confidence Threshold
                                         0.65, # IoU Threshold
                                         nc=self.model.yaml['nc'], # Number of Classes
                                         nkpt=self.model.yaml['nkpt'], # Number of Keypoints
                                         kpt_label=True)

        with torch.no_grad():
            output = output_to_keypoint(output)

        nimg = image[0].permute(1, 2, 0) * 255
        nimg = nimg.cpu().numpy().astype(np.uint8)
        nimg = cv2.cvtColor(nimg, cv2.COLOR_RGB2BGR)
        for idx in range(output.shape[0]):
            plot_skeleton_kpts(nimg, output[idx, 7:].T, 3)
        for personidx, person in enumerate(output):
            cx = person [2]
            cy =  person [3]
            bboxw =  person [4]
            bboxh =  person [5]
            x0 = cx - (bboxw/2)
            y0 = cy - (bboxh/2)
            x1 = x0 + bboxw
            y1 = y0 + bboxh
            output[personidx][2:6] = [x0,y0,x1,y1]
        return nimg[:,:,::-1], output



if __name__ == "__main__":
    yolopose = yolopose()
    output, image = yolopose.run_inference(cv2.imread('b.jpg')) # Bryan Reyes on Unsplash
    viz = yolopose.visualize_output(output, image)
    cv2.imshow("",viz)
    cv2.waitKey(0)
