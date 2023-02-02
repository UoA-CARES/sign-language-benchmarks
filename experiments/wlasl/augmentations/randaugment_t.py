from ..builder import PIPELINES
import numpy as np
import random
import PIL
import PIL.ImageOps
import PIL.ImageEnhance
import PIL.ImageDraw
from PIL import Image


def temporal_interpolate(v_list, t, n):
    if len(v_list) == 1:
        return v_list[0]
    elif len(v_list) == 2:
        return v_list[0] + (v_list[1] - v_list[0]) * t / n
    else:
        NotImplementedError('Invalid degree')


class Augment:
    def __init__(self):
        pass

    def __call__(self, buffer):
        raise NotImplementedError

    def ShearX(self, imgs, v_list):  # [-0.3, 0.3]
        for v in v_list:
            assert -0.3 <= v <= 0.3
        if random.random() > 0.5:
            v_list = [-v for v in v_list]

        out = [img.transform(img.size, PIL.Image.Transform.AFFINE, (1, temporal_interpolate(
            v_list, t, len(imgs) - 1), 0, 0, 1, 0)) for t, img in enumerate(imgs)]
        return out

    def ShearY(self, imgs, v_list):  # [-0.3, 0.3]
        for v in v_list:
            assert -0.3 <= v <= 0.3
        if random.random() > 0.5:
            v_list = [-v for v in v_list]

        out = [img.transform(img.size, PIL.Image.Transform.AFFINE, (1, 0, 0, temporal_interpolate(
            v_list, t, len(imgs) - 1), 1, 0)) for t, img in enumerate(imgs)]
        return out

    # [-150, 150] => percentage: [-0.45, 0.45]
    def TranslateX(self, imgs, v_list):
        for v in v_list:
            assert -0.45 <= v <= 0.45
        if random.random() > 0.5:
            v_list = [-v for v in v_list]
        v_list = [v * imgs.size[1] for v in v_list]

        out = [img.transform(img.size, PIL.Image.Transform.AFFINE, (1, 0, temporal_interpolate(
            v_list, t, len(imgs) - 1), 0, 1, 0)) for t, img in enumerate(imgs)]
        return out

    # [-150, 150] => percentage: [-0.45, 0.45]
    def TranslateXabs(self, imgs, v_list):
        for v in v_list:
            assert 0 <= v
        if random.random() > 0.5:
            v_list = [-v for v in v_list]

        out = [img.transform(img.size, PIL.Image.Transform.AFFINE, (1, 0, temporal_interpolate(
            v_list, t, len(imgs) - 1), 0, 1, 0)) for t, img in enumerate(imgs)]
        return out

    # [-150, 150] => percentage: [-0.45, 0.45]
    def TranslateY(self, imgs, v_list):
        for v in v_list:
            assert -0.45 <= v <= 0.45
        if random.random() > 0.5:
            v_list = [-v for v in v_list]
        v_list = [v * imgs.size[2] for v in v_list]

        out = [img.transform(img.size, PIL.Image.Transform.AFFINE, (1, 0, 0, 0, 1, temporal_interpolate(
            v_list, t, len(imgs) - 1))) for t, img in enumerate(imgs)]
        return out

    # [-150, 150] => percentage: [-0.45, 0.45]
    def TranslateYabs(self, imgs, v_list):
        for v in v_list:
            assert 0 <= v
        if random.random() > 0.5:
            v_list = [-v for v in v_list]

        out = [img.transform(img.size, PIL.Image.Transform.AFFINE, (1, 0, 0, 0, 1, temporal_interpolate(
            v_list, t, len(imgs) - 1))) for t, img in enumerate(imgs)]
        return out

    def Rotate(self, imgs, v_list):  # [-30, 30]
        for v in v_list:
            assert -30 <= v <= 30
        if random.random() > 0.5:
            v_list = [-v for v in v_list]

        out = [img.rotate(temporal_interpolate(v_list, t, len(imgs) - 1))
               for t, img in enumerate(imgs)]
        return out

    def AutoContrast(self, imgs, _):
        out = [PIL.ImageOps.autocontrast(img) for img in imgs]
        return out

    def Invert(self, imgs, _):
        out = [PIL.ImageOps.invert(img) for img in imgs]
        return out

    def Equalize(self, imgs, _):
        out = [PIL.ImageOps.equalize(img) for img in imgs]
        return out

    def Flip(self, imgs, _):  # not from the paper
        out = [PIL.ImageOps.mirror(img) for img in imgs]
        return out

    def Solarize(self, imgs, v_list):  # [0, 256]
        for v in v_list:
            assert 0 <= v <= 256

        out = [PIL.ImageOps.solarize(img, temporal_interpolate(
            v_list, t, len(imgs) - 1)) for t, img in enumerate(imgs)]
        return out

    def Posterize(self, imgs, v_list):  # [4, 8]
        v_list = [max(1, int(v)) for v in v_list]
        v_list = [max(1, int(v)) for v in v_list]

        out = [PIL.ImageOps.posterize(img, int(temporal_interpolate(
            v_list, t, len(imgs) - 1))) for t, img in enumerate(imgs)]
        return out

    def Contrast(self, imgs, v_list):  # [0.1,1.9]
        for v in v_list:
            assert 0.1 <= v <= 1.9

        out = [PIL.ImageEnhance.Contrast(img).enhance(temporal_interpolate(
            v_list, t, len(imgs) - 1)) for t, img in enumerate(imgs)]
        return out

    def Color(self, imgs, v_list):  # [0.1,1.9]
        for v in v_list:
            assert 0.1 <= v <= 1.9

        out = [PIL.ImageEnhance.Color(img).enhance(temporal_interpolate(
            v_list, t, len(imgs) - 1)) for t, img in enumerate(imgs)]
        return out

    def Brightness(self, imgs, v_list):  # [0.1,1.9]
        for v in v_list:
            assert 0.1 <= v <= 1.9

        out = [PIL.ImageEnhance.Brightness(img).enhance(temporal_interpolate(
            v_list, t, len(imgs) - 1)) for t, img in enumerate(imgs)]
        return out

    def Sharpness(self, imgs, v_list):  # [0.1,1.9]
        for v in v_list:
            assert 0.1 <= v <= 1.9

        out = [PIL.ImageEnhance.Sharpness(img).enhance(temporal_interpolate(
            v_list, t, len(imgs) - 1)) for t, img in enumerate(imgs)]
        return out

    def Identity(self, imgs, _):
        return imgs

    def augment_list(self):
        # list of data augmentations and their ranges
        l = [
            (self.Identity, 0, 1),
            (self.AutoContrast, 0, 1),
            (self.Equalize, 0, 1),
            (self.Invert, 0, 1),
            (self.Rotate, 0, 30),
            (self.Posterize, 0, 4),
            (self.Solarize, 0, 256),
            (self.Color, 0.1, 1.9),
            (self.Contrast, 0.1, 1.9),
            (self.Brightness, 0.1, 1.9),
            (self.Sharpness, 0.1, 1.9),
            (self.ShearX, 0., 0.3),
            (self.ShearY, 0., 0.3),
            (self.TranslateXabs, 0., 100),
            (self.TranslateYabs, 0., 100),
        ]

        return l


@PIPELINES.register_module()
class RandAugment_T(Augment):
    """Apply temporal RandAugment to the video.

    See paper "Learning Temporally Invariant and Localizable Features via
    Data Augmentation for Video Recognition", Taeoh Kim et al., 2020
    (https://arxiv.org/pdf/2008.05721.pdf) for details.

    Args:
        n (int): Number of augments to be applied sequentially. Default: 2.
        m (int): Magnitude of each augment between range [0,30]. Default: 7.
    """

    def __init__(self, n=2, m=7, temp_degree=1, range=1.0):
        super(RandAugment_T, self).__init__()
        self.max_severity = 30
        self.temp_degree = temp_degree
        self.n = n
        self.m = m  # usually values in the range [5, 30] is best
        self.range = range
        self.augment_list = self.augment_list()

    def __call__(self, results):
        buffer = [Image.fromarray(img.astype('uint8'))
                  for img in np.array(results['imgs'])]

        ops = random.choices(self.augment_list, k=self.n)
        for op, minval, maxval in ops:
            if self.temp_degree == 0:
                val_list = [(float(self.m) / self.max_severity)
                            * float(maxval - minval) + minval]
            else:  # temp_degree == 1
                tval = float(np.random.uniform(
                    low=0.0, high=0.5 * self.range * self.m))
                if random.random() > 0.5:
                    val_list = [((float(self.m) - tval) / self.max_severity)
                                * float(maxval - minval) + minval]
                    val_list.extend(
                        [((float(self.m) + tval) / self.max_severity) * float(maxval - minval) + minval])
                else:
                    val_list = [((float(self.m) + tval) / self.max_severity)
                                * float(maxval - minval) + minval]
                    val_list.extend(
                        [((float(self.m) - tval) / self.max_severity) * float(maxval - minval) + minval])
            buffer = op(buffer, val_list)

        results['imgs'] = np.array(
            [np.array(img, np.dtype('int64')) for img in buffer])

        return results
