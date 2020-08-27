# pylint: disable=W0223
# Details : https://github.com/PyCQA/pylint/issues/3098

"""
Base module for all video url handlers
"""
from abc import ABC
import io
import torch
import numpy
import concurrent.futures
import math
import torchvision.io as io
import torchvision
from PIL import Image
from .base_handler import BaseHandler
from ..utils import read_video

class VideoHandler(BaseHandler, ABC):
    """
    Base class for all vision handlers
    """
    def preprocess(self, data):
        video_streams = []
        for row in data:
            data = row.get("data") or row.get("body")
            with concurrent.futures.ProcessPoolExecutor() as executor:
                results = []
                #tmp add 5 frame at here
                for result in zip(executor.map(read_video, [(data, i , 5) for i in range(5)])):
                    results.append(result[0][0])
            video_streams.append(torch.stack(results))

        return torch.stack(images)
