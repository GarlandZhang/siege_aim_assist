from PIL import ImageGrab
import numpy as np
import cv2
import time

def get_computer_image(box=(0, 0, 1024, 768)):
  start_time = time.time()
  img = ImageGrab.grab(box)
  arr = np.array(img)
  end_time = time.time()
  print('Image extraction time: {}'.format(end_time - start_time))
  return arr


