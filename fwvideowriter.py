#
#
#
#

import cv2 as cv
from enum import Enum, auto

class VideoCodec(Enum):
  H264 = auto()
  MP4  = auto()
  AVI  = auto()

class FWVideoWriter:
  def __init__(self, filename, framerate, shape, videocodec):
    self.filename = filename
    self.framerate = framerate
    self.shape = shape
    vwres = tuple(self.resolution)
    if videocodec == VideoCodec.H264: 
      self.writer = cv.VideoWriter(filename+'.mkv',cv.VideoWriter_fourcc(*'x264'),framerate,vwres)
    elif videocodec == VideoCodec.MP4:
      self.writer = cv.VideoWriter(filename+'.mp4',cv.VideoWriter_fourcc(*'MP4V'),framerate,vwres)
    elif videocodec == VideoCodec.AVI:
      self.writer = cv.VideoWriter(filename+'.avi',cv.VideoWriter_fourcc(*'MJPG'),framerate,vwres)
    else:
      raise ValueError('Invalid videocodec value')

  @classmethod
  def with_specs(cls, filename, shape=(720,1280), framerate=24, videocodec=VideoCodec.H264):
    return cls(filename, framerate, shape, videocodec)

  def get_resolution(self):
    return self.shape[::-1]

  resolution = property(get_resolution)

  def write(self, frame):
    self.writer.write(frame) 

  def __del__(self):
    self.writer.release()

