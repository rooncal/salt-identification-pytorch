import torchvision.transforms.functional as F
import torchvision.transforms as T



class RandomRotate(object):
  def __call__(self, img, mask):
    if random.random() > 0.5:
      random_angle = random.uniform(0, 0.3)
      img = F.rotate(img, angle=random_angle)
      mask = F.rotate(mask, angle=random_angle)
    return img, mask

class RandomHorizontalFlip(object):
  def __call__(self, img, mask):
    if random.random() > 0.5:
      img = F.hflip(img)
      mask = F.hflip(mask)
    return img, mask

class RandomVerticalFlip(object):
  def __call__(self, img, mask):
    if random.random() > 0.5:
      img = F.vflip(img)
      mask = F.vflip(mask)
    return img, mask

class ToTensor(object):
  def __call__(self, img, mask):
    return F.to_tensor(img), F.to_tensor(mask)

class Resize(object):
  def __init__(self, size=(128, 128)):
    self.size = size

  def __call__(self, img, mask):
    resize = T.Resize(size=self.size)
    return resize(img), resize(mask)

class Normalize(object):
  def __init__(self, img_mean, img_std):
    self.img_mean = img_mean
    self.img_std = img_std

  def __call__(self, img, mask):
    return F.normalize(img, self.img_mean, self.img_std), mask // 65335

class Normalize_Mask(object):
  def __call__(self, img, mask):
    return img, mask // 65335

class Compose(T.Compose):
    def __call__(self, img, mask):
        for t in self.transforms:
            img, mask = t(img, mask)
        return img, mask