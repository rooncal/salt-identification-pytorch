from PIL import Image
from glob import glob
import os

class DataSetSegmentation(data.Dataset):
    def __init__(self, images_path, masks_path, transform=None):
        super(DataSetSegmentation, self).__init__()
        self.img_files = glob(os.path.join(images_path, '*.png'))
        self.mask_files = []
        for img in self.img_files:
           self.mask_files.append(os.path.join(masks_path, os.path.basename(img)))
        self.transform = transform
 
    def __getitem__(self, index):
            img_path = self.img_files[index]
            mask_path = self.mask_files[index]
            data = Image.open(img_path).convert('RGB')
            label = Image.open(mask_path)
            if self.transform:
              transformed_data, transformed_label = self.transform(data, label)
              return transformed_data, transformed_label.float()
            return data, label
 
    def __len__(self):
        return len(self.img_files)