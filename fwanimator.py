#
#
#
#

from .fwvideowriter import *
from .fwcanvas import *
import math
from enum import Enum, auto
import numpy as np

def _f(t):
  return 0.5*(math.tanh(8*t - 4) + 1) 

def _h(t):
  return _f(0)*_f(t)

def curve(t):
  return (1./_h(1.))*_h(t)
  
def smoothly(t, v0, v1):
  v = curve(t)
  return v*v1 + (1 - v)*v0

def _convert_color(color):
  if isinstance(color, str):
    r = int(color[ :2], 16)
    g = int(color[2:4], 16)
    b = int(color[4:6], 16)
    return (r, g, b)
  else:
    return color

class FWQuality(Enum):
  TEST = auto()
  PRODUCTION = auto()

class FWAnimator:
  def __init__(self, ratio, scale, framerate, filename, videocodec, defaultcolor):
    self.ratio = ratio
    self.scale = scale
    self.size  = ratio*scale
    self.framerate = framerate
    self.writer = FWVideoWriter.with_specs(filename, self.size, framerate, videocodec)
    self.canvas = FWCanvas.with_shape(self.size, defaultcolor)
    self.set()

  @classmethod
  def test_animator(cls, filename='test', defaultcolor=(25, 6, 17)):
    return cls(np.array((9, 16)), 80, 24, filename, VideoCodec.H264, _convert_color(defaultcolor))

  @classmethod
  def production_animator(cls, filename='test', defaultcolor=(25, 6, 17)):
    return cls(np.array((9, 16)), 240, 60, filename, VideoCodec.AVI, _convert_color(defaultcolor))

  @classmethod
  def with_quality(cls, quality, filename='test', defaultcolor=(25, 6, 17)):
    if quality == FWQuality.TEST:
      return FWAnimator.test_animator(filename, _convert_color(defaultcolor))
    elif quality == FWQuality.PRODUCTION:
      return FWAnimator.production_animator(filename, _convert_color(defaultcolor))

  @classmethod
  def custom_animator(cls, ratio, scale, framerate, filename, videocodec, defaultcolor):
    return cls(ratio, scale, framerate, filename, videocodec, _convert_color(defaultcolor))

  def get_resolution(self):
    return self.size[::-1]

  def get_center(self):
    return self.ratio/2

  resolution = property(get_resolution)
  center = property(get_center)

  def next_frame(self):
    self.writer.write(self.canvas.frame)

  def line(self, p0, p1, color=(255, 255, 255), stroke=0.05, continued=False, final=True):
    a_p0 = p0 * self.scale
    a_p1 = p1 * self.scale
    a_stroke = stroke * self.scale
    self.canvas.draw_line(a_p0, a_p1, _convert_color(color), a_stroke, continued, final)

  def rel_line(self, dp, color=(255, 255, 255), stroke=0.05, final=True):
    a_dp = dp * self.scale
    a_stroke = stroke * self.scale
    self.canvas.draw_rel_line(a_dp, _convert_color(color), a_stroke, continued, final)

  def curve(self, p0, c0, c1, p1, color=(255, 255, 255), stroke=0.05, continued=False, final=True):
    a_p0 = p0 * self.scale
    a_c0 = c0 * self.scale
    a_c1 = c1 * self.scale
    a_p1 = p1 * self.scale
    a_stroke = stroke * self.scale
    self.canvas.draw_curve(a_p0, a_c0, a_c1, a_p1, _convert_color(color), a_stroke, continued, final)

  def rel_curve(self, dc0, dc1, dp1, color=(255, 255, 255), stroke=0.05, final=True):
    a_dc0 = dc0 * self.scale
    a_dc1 = dc1 * self.scale
    a_dp1 = dp1 * self.scale
    a_stroke = stroke * self.scale
    self.canvas.draw_rel_curve(a_dc0, a_dc1, a_dp1, _convert_color(color), a_stroke, continued, final)

  def circle(self, center, radius, color=(255, 255, 255), fill=True, stroke=5):
    a_center = center*self.scale
    a_radius = radius*self.scale
    a_stroke = stroke*self.scale
    self.canvas.draw_circle(a_center, a_radius, _convert_color(color), fill, a_stroke)

  def rect(self, center, size, color, fill=True, stroke=5):
    a_center = center*self.scale
    a_size   = size*self.scale
    a_stroke = stroke*self.scale
    self.canvas.draw_rect(a_center, a_size, _convert_color(color), fill, a_stroke)

  def text(self, text, font, origin, size, color, positioning=FontPositioning.CENTER):
    a_origin = origin*self.scale
    a_size   = size*self.scale
    a_size = self.canvas.add_text(text, font, a_origin, a_size, _convert_color(color), positioning)

  def image(self, image, center=-1, size=np.array((-1, 1)), interp=Interpolation.CUBIC, cache=True):
    if not isinstance(center, int):
      a_center = center * self.scale
    else:
      a_center = -1

    a_size = np.array((0, 0))
    if size[0] < 0:
      a_size[1] = self.scale * size[1]
      a_size[0] = image.shape[0] * a_size[1] / image.shape[1]
    elif size[1] < 0:
      a_size[0] = self.scale * size[0]
      a_size[1] = image.shape[1] * a_size[0] / image.shape[0]
    else:
      a_size = size * self.scale
    self.canvas.draw_image(image, a_center, a_size, interp, cache)
    return a_size/self.scale

  def set(self, color=-1):
    self.canvas.set(_convert_color(color))

  def a_to_r(self, vec):
    return vec / self.scale

  def r_to_a(self, vec):
    return vec * self.scale

  def __del__(self):
    del self.writer

