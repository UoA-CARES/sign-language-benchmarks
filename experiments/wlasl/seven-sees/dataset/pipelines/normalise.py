# MMAction2 2023

import numpy as np
import torch


class Normalise:
    """Normalises frames in a video.

    Required keys are "mean" and "std".
    Args:
        mean (Sequence[float]): The mean of the channels.
        std (Sequence[float]): The standard deviation of the channels.
    """

    def __init__(self,
                 mean,
                 std
                 ):

        self.mean = np.array(mean).reshape(-1, 1, 1)
        self.std = np.array(std).reshape(-1, 1, 1)

    def normalise(self, image):
        """Perform normalise on a single frame.
        Args:
            image (np.array): The image as a numpy array.
        Returns:
            image (np.array): The normalised image as numpy array.
        """
        stdinv = 1/self.std

        image = image-self.mean
        image = image*stdinv

        return image

    def __call__(self, images):
        """Perform normalise on the video.
        Args:
            images (Tensor): The frames of the video in CNHW format.
        Returns:
            out (Tensor): The normalised frames in CNHW format.
        """
        img_array = np.array(images.permute(1, 0, 2, 3))

        n = len(img_array)
        c, h, w = img_array[0].shape
        img_ = np.empty((n, c, h, w))

        for i, img in enumerate(img_array):
            img_[i] = self.normalise(img)

        out = torch.tensor(img_)

        return out.permute(1, 0, 2, 3)
