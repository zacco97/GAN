from torch.utils.data import Dataset
import glob
import pandas as pd
import os
from cv2 import imread, cvtColor, COLOR_BGR2RGB
from matplotlib import pyplot as plt

class DatasetMonet(Dataset):
    def __init__(self, root_dir:str= "", target_dir:str="", transform=None):
        self.root_dir = root_dir
        self.target_dir = target_dir
        self.transform = transform
        self.df = self.make_csv()
    
    def __len__(self):
        return len(self.df)

    # def __str__(self):
    #     return f"{self.df.head(5).to_dict()}"
        
    def __getitem__(self, index):
        img = imread(self.df.loc[index, "image"])
        img_monet = imread(self.df.loc[index, "image_monet"])
        
        im = cvtColor(img, COLOR_BGR2RGB)
        im_monet = cvtColor(img_monet, COLOR_BGR2RGB)
        
        if self.transform:
            sample = self.transform(image=im, image0=im_monet)
            im, im_monet = sample["image"], sample["image0"]
        return im, im_monet
    
    def visualize(self, idx):
        ret = self.__getitem__(idx)
        fig, (ax1, ax2) = plt.subplots(1, 2)
        ax1.imshow(ret[0].detach().numpy().transpose(1,2,0))
        ax2.imshow(ret[1].detach().numpy().transpose(1,2,0))
        plt.show()
    
    def make_csv(self):
        images = []
        images_monet = []
        targets_images = glob.glob(os.path.join(self.target_dir, "*.jpg"))
        for i, file in enumerate(os.listdir(self.root_dir)):
            images.append(os.path.join(self.root_dir, file))
            idx_monet = i % len(targets_images)
            images_monet.append(targets_images[idx_monet]) 
        
        df = pd.DataFrame({"image":images, "image_monet":images_monet})
        print(df)
        return df