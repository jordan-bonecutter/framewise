#
#
#
#

import numpy as np
import cv2 as cv
import cairo
from enum import Enum, auto
import math

class FontSlant(Enum):
  '''
  Enum wrapper for cairo.FontSlant. Has values:
    NORMAL
    ITALIC
    OBLIQUE
  '''
  NORMAL  = cairo.FontSlant.NORMAL
  ITALIC  = cairo.FontSlant.ITALIC
  OBLIQUE = cairo.FontSlant.OBLIQUE

class FontWeight(Enum):
  '''
  Enum wrapper for cairo.FontWeight. Has values:
    NORMAL
    BOLD
  '''
  NORMAL = cairo.FontWeight.NORMAL
  BOLD   = cairo.FontWeight.BOLD

class FontPositioning(Enum):
  CENTER        = auto()
  TOP_LEFT      = auto()
  TOP_RIGHT     = auto()
  TOP_CENTER    = auto()
  BOTTOM_LEFT   = auto()
  BOTTOM_RIGHT  = auto()
  BOTTOM_CENTER = auto()
  RIGHT_CENTER  = auto()
  LEFT_CENTER   = auto()

class Interpolation(Enum):
  NEAREST = cv.INTER_NEAREST
  LINEAR  = cv.INTER_LINEAR
  AREA    = cv.INTER_AREA
  CUBIC   = cv.INTER_CUBIC

class FWCanvas:
  '''
  The FWCanvas object interfaces with lower level modules like
  cairo for drawing and opencv for images. It does not need to
  be instantiated by the user as the FWVideo maintains its own
  FWCanvas object. However, if you want to the easiest way
  would be to use:

    FWCanvas.with_shape(shape)

  where shape is a tuple in the format (height, width) the same
  as a numpy array. You can also instaitate an FWCanvas using 
  your own numpy array. The numpy array must have shape
  (height, width, 4) and dtype numpy.uint8. 
  '''

  def __init__(self, internal, defaultcolor=(25, 6, 17)):
    '''
    The init method for the FWCanvas sets up the backend support
    for drawing with cairo as well as setting up some internal
    variables. This should not be called by the user, instead 
    the user should call FWCanvas.with_shape or 
    FWCanvas.with_array()
    '''
    self.internal = internal
    self.defaultcolor = np.array(list(defaultcolor) + [255], dtype=np.uint8)
    rx, ry            = self.resolution
    surface           = cairo.ImageSurface.create_for_data(self.internal, cairo.Format.RGB24, rx, ry)
    self.context      = cairo.Context(surface)
    self.fonts        = {}
    self.imagecache   = {}

  @classmethod
  def with_shape(cls, shape, defaultcolor=(25, 6, 17)):
    '''
    Creates a new FWCanvas with height, width = shape 
    '''
    if len(shape) != 2:
      raise ValueError('Shape must be two dimensional in the format (height, width).')
    return cls(np.zeros(tuple(list(shape) + [4]), dtype=np.uint8), defaultcolor)

  @classmethod
  def with_array(cls, array, defaultcolor=(25, 6, 17)):
    '''
    Creates a new FWCanvas with np.ndarrary
    '''
    if not isinstance(array, np.ndarray):
      raise ValueError('Array must be a numpy.ndarray')
    if array.dtype != np.uint8:
      raise ValueError('Array must have data type numpy.uint8')
    if len(array.shape) != 3:
      raise ValueError('Array must have shape (X, Y, 4)')
    elif array.shape[2] != 4:
      raise ValueError('Array must have shape (X, Y, 4)')

    return cls(array, defaultcolor)

  def get_frame(self):
    return self.internal[:, :, :3]

  def get_shape(self):
    '''
    Getter for shape. Format is (height, width)
    '''
    return self.internal.shape[:2] 

  def get_resolution(self):
    '''
    Getter for resolution. Format is (width, height)
    which is more typical for consumer applications
    '''
    return (self.internal.shape[1], self.internal.shape[0])

  frame = property(get_frame)
  shape = property(get_shape)
  resolution = property(get_resolution)

  def draw_line(self, p0, p1, color=(255, 255, 255), stroke=5, continued=False, final=True):
    if not continued:
      self.context.move_to(p0[1], p0[0])

    self.context.line_to(p1[1], p1[0]) 

    if final:
      self.context.set_line_width(stroke)
      self.context.set_source_rgb(color[0]/255, color[1]/255, color[2]/255)
      self.context.stroke()

  def draw_rel_line(self, dp, color=(255, 255, 255), stroke=5, final=True):
    self.context.rel_line_to(dp[1], dp[0]) 

    if final:
      self.context.set_line_width(stroke)
      self.context.set_source_rgb(color[0]/255, color[1]/255, color[2]/255)
      self.context.stroke()

  def draw_curve(self, p0, c0, c1, p1, color=(255, 255, 255), stroke=5, continued=False, final=True):
    if not continued:
      self.context.move_to(p0[1], p0[0])

    self.context.curve_to(c0[1], c0[0], c1[1], c1[0], p1[1], p1[0]) 

    if final:
      self.context.set_line_width(stroke)
      self.context.set_source_rgb(color[0]/255, color[1]/255, color[2]/255)
      self.context.stroke()

  def draw_rel_curve(self, dc0, dc1, dp1, color=(255, 255, 255), stroke=5, final=True):
    self.context.curve_to(dc0[1], dc0[0], dc1[1], dc1[0], dp1[1], dp1[0]) 

    if final:
      self.context.set_line_width(stroke)
      self.context.set_source_rgb(color[0]/255, color[1]/255, color[2]/255)
      self.context.stroke()

  def draw_circle(self, center, radius, color=(255, 255, 255), fill=True, stroke=5):
    '''
    Draw a circle with center at y_c = xenter[0], x_c = center[1].
    If fill is set to True, the circle will be filled with color.
    Otherwise, the circle will be outlined with a stroke width of
    stroke.
    '''
    self.context.arc(center[1], center[0], radius, 0, math.tau)
    self.context.set_source_rgb(color[0]/255, color[1]/255, color[2]/255)
    if fill:
      self.context.fill()
    else:
      self.context.set_line_width(stroke)
      self.context.stroke()

  def draw_rect(self, center, size, color, fill=True, stroke=5):
    '''
    Draw a rectangle with center at y_c = xenter[0], x_c = center[1].
    If fill is set to True, the rectangle will be filled with color.
    Otherwise, the rectangle will be outlined with a stroke width of
    stroke.
    '''
    self.context.rectangle(center[1] - size[1]/2, center[0] - size[0]/2, size[1], size[0])
    self.context.set_source_rgb(color[0]/255, color[1]/255, color[2]/255)
    if fill:
      self.context.fill()
    else:
      self.context.set_line_width(stroke)
      self.context.stroke()

  def add_text(self, text, fontname, origin, size, color, positioning=FontPositioning.CENTER):
    '''
    Add text with font of name fontname. If centered=True, then
    the origin denotes the center of the text to be drawn. Otherwise
    it respresents the top left corner. If target=FontSizing.WIDTH, then
    size represents the width in pixels for the target text. Similarly
    for height. Otherwise if target=FontSizing.SCALE the size represents
    the font size factor passed to cairo.
    '''
    # get font
    try:
      font = self.fonts[fontname]
    except KeyError:
      font = cairo.ToyFontFace(fontname)
      self.fonts[fontname] = font

    # create options
    opt = cairo.FontOptions()

    # get size for width or height
    testfont = cairo.ScaledFont(font, cairo.Matrix(1, 0, 0, 1, 0, 0), cairo.Matrix(1, 0, 0, 1, 0, 0), opt)
    extents = testfont.text_extents(text)
    try:
      size = size / extents.height
    except ZeroDivisionError:
      return

    # create font for drawing 
    usefont = cairo.ScaledFont(font, cairo.Matrix(size,0,0,size,0,0), cairo.Matrix(1,0,0,1,0,0), opt)
    if positioning == FontPositioning.CENTER:
      extents = usefont.text_extents(text)
      origin[0] += extents.height/2
      origin[1] -= extents.width/2
      origin[1] -= extents.x_bearing
    elif positioning == FontPositioning.TOP_LEFT:
      extents = usefont.text_extents(text)
      origin[0] += extents.height
    elif positioning == FontPositioning.TOP_RIGHT:
      extents = usefont.text_extents(text)
      origin[0] += extents.height
      origin[1] -= extents.width
      origin[1] -= extents.x_bearing
    elif positioning == FontPositioning.TOP_CENTER:
      extents = usefont.text_extents(text)
      origin[0] += extents.height
      origin[1] -= extents.width/2
      origin[1] -= extents.x_bearing
    elif positioning == FontPositioning.BOTTOM_RIGHT:
      extents = usefont.text_extents(text)
      origin[1] -= extents.width
      origin[1] -= extents.x_bearing
    elif positioning == FontPositioning.BOTTOM_CENTER:
      extents = usefont.text_extents(text)
      origin[1] -= extents.width/2
      origin[1] -= extents.x_bearing
    elif positioning == FontPositioning.RIGHT_CENTER:
      extents = usefont.text_extents(text)
      origin[0] += extents.height/2
      origin[1] -= extents.width
      origin[1] -= extents.x_bearing
    elif positioning == FontPositioning.LEFT_CENTER:
      extents = usefont.text_extents(text)
      origin[0] += extents.height/2

    self.context.set_source_rgb(color[0]/255, color[1]/255, color[2]/255)
    self.context.set_scaled_font(usefont)
    self.context.move_to(origin[1], origin[0])
    self.context.show_text(str(text))
    self.context.fill()

  def draw_image(self, image, center=-1, size=np.array((1, 1)), interp=Interpolation.CUBIC, cache=True):
    '''
    Draw an image centered at center. If no center is specified, it
    will deafult to the center of the canvas. The image may also be
    scaled by a factor of scale.
    '''
    sy, sx = int(size[0]), int(size[1])
    if cache:
      key = (id(image), sy, sx)
      
      # cache hit
      if key in self.imagecache:
        image = self.imagecache[key]

      # cache miss
      else:
        if (sy, sx) != image.shape[:2]:
          image = cv.resize(image, (sx, sy), interpolation=interp.value)
        self.imagecache[key] = image

    else:
      image = cv.resize(image, (sx, sy), interpolation=interp.value)

    if isinstance(center, int) and center == -1:
      center = (self.internal.shape[0]/2, self.internal.shape[1]/2)

    y0 = int(center[0] - image.shape[0]/2)
    y1 = y0 + image.shape[0]
    x0 = int(center[1] - image.shape[1]/2)
    x1 = x0 + image.shape[1]

    yoffset = 0
    if y0 < 0:
      yoffset = -y0
      y0 = 0
    xoffset = 0
    if x0 < 0:
      xoffset = -x0
      x0 = 0
    
    y0 = min(y0, self.shape[0])
    y1 = max(0, min(y1, self.shape[0]))
    x0 = min(x0, self.shape[1])
    x1 = max(0, min(x1, self.shape[1]))

    self.internal[y0:y1, x0:x1, :3] = image[yoffset:yoffset+y1-y0, xoffset:xoffset+x1-x0, :3]

  def set(self, clearcolor=-1):
    '''
    Clears the canvas to its default color unless clearcolor is specified 
    '''
    if isinstance(clearcolor, int):
      self.internal[:, :] = self.defaultcolor
    elif isinstance(clearcolor, tuple):
      if len(clearcolor) != 3:
        raise ValueError('Color must be a 3 tuple.')
      clearcolor = np.array(list(clearcolor) + [255], dtype=np.uint8)
      self.internal[:, :] = clearcolor

