import os, cv2
import numpy as np

def load_image(path, image_size: tuple=None):
  image = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_UNCHANGED) # np.fromFile supports reading filepaths with unicode.
  image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  if image_size is not None: 
    image = cv2.resize(image, (image_size[1], image_size[0]))
  return image

def load_image_dataset(path, image_size: tuple=None, normalize=False):
  for root, dirs, files in os.walk(path):
    if normalize:
      return (files, np.array([load_image(path / file, image_size) for file in files])/255.0)
    return (files, np.array([load_image(path / file, image_size) for file in files]))