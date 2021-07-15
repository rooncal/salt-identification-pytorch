import torch.utils.data as data
from PIL import Image

class BinaryDataset(data.Dataset):
    def __init__(self, train_df, image_folder="/content/data",transform=None):
        self.images = image_folder
        self.df = train_df
        self.transform = transform
    
    def __len__(self):
        return len(os.listdir(image_folder))
    
    def __getitem__(self, idx):
        name = self.df.id[idx] + '.png'
        image = Image.open(os.path.join(self.images, name)).convert('RGB')
        if self.transform:
            image = self.transform(image)
        val = self.df.binary[idx]
        #val = np.array([val], dtype="long")
    
        #sample = {'image': image, 'val': val}
        return (image, val)