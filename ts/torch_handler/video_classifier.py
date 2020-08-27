"""
Module for image classification default handler
"""
import torch
import torch.nn.functional as F
from torchvision import transforms

from .video_handler import VideoHandler
from ..utils.util  import map_class_to_label


class VideoClassifier(VideoHandler):
    """
    ImageClassifier handler class. This handler takes an image
    and returns the name of object in that image.
    """

    topk = 1
    # These are the standard Imagenet dimensions
    # and statistics
    softmax = F.Softmax(dim=-1)

    normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            normalize
        ])


    def preprocess(self, data):
        """
        x: NFCHW: tensor CPU; C = 3: RGB order
        return: list of str ['Game', 'Cartoo', 'Other (Entertainment)'] (len N)
        """
        x = super().preprocess(data)
        bs, n_frame = x.size(0), x.size(1)
        x = torch.squeeze(x[:, n_frame // 2, ...])
        for i in range(bs):
            x_ = transform(x[i, ...])[None, ...]
            return x_



    def set_max_result_classes(self, topk):
        self.topk = topk

    def get_max_result_classes(self):
        return self.topk

    def postprocess(self, data):
        ps = softmax(game_pred).data.cpu().numpy()[:, 1]
        probs, classes = torch.topk(ps, self.topk, dim=1)
        probs = probs.tolist()
        classes = classes.tolist()
        return map_class_to_label(probs, self.mapping, classes)
