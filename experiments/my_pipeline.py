from ..builder import PIPELINES
import json
import numpy as np

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


@PIPELINES.register_module()
class MyTransform(object):
    def __call__(self, results):
        results['imgs'] = list(np.array(results['imgs']))
        return results
    

@PIPELINES.register_module()
class SaveContents(object):
    def __call__(self, results):
        if 'written' not in results:
            results['written'] = False

        if not results['written']:
            with open('results.json', 'w') as convert_file:
                convert_file.write(json.dumps(results, cls=NpEncoder))
            results['written'] = True

        # print(type(results['imgs']))
        return results

@PIPELINES.register_module()
class CutOut(object):
    """Cut out a portion of an image to black.

    Required key is "box_size".

    Args:
        box_size (int): The side length of the box.
    """

    def __init__(self, box_size):
        self.box_size = box_size

    def __call__(self, results):
        imgs = np.array(results['imgs'])
        assert self.box_size < imgs.shape[1]
        box_h, box_w = random.randint(0, imgs.shape[1]-self.box_size), random.randint(0, imgs.shape[2]-self.box_size)
        imgs[:, box_h:box_h+self.box_size, box_w:box_w+self.box_size, :] = 0
        results['imgs'] = list(imgs)
        return results