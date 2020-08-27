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

def read_video(args):
    """
    split video and read specific frame
    :param args->tuple(content, nfrmae, maxframe)
    :return: nth frame of videos
    """
    content, nframe, maxframe = args
    info = io._probe_video_from_memory(content)
    frament = float(info['video_duration'])/ (maxframe + 1) * nframe
    start_offset = int(math.floor(frament * (1 / info["video_timebase"])))
    end_offset = start_offset + int(info["video_timebase"].denominator / info['video_duration'].denominator)
    return io._read_video_from_memory(content, read_audio_stream = 0,
            video_pts_range=[start_offset, end_offset])
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
