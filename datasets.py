import torch
import numpy as np
import cv2
import PIL.Image as Image

class CatDataset(torch.utils.data.Dataset):
    def __init__(self,
                 reshape_size=[360,360,3],
                 interval=[-1,1],
                 device="cuda:0",
                 dtype=torch.float32,
                 random_mask_augment=False,
                 split_name="training",
                 data_path="./MissingDataOpenData/MissingDataOpenData/"):
        self.reshape_size = reshape_size
        self.random_mask_augment = random_mask_augment
        self.device = device
        self.interval = interval
        self.data_path = data_path
        self.filenames = np.loadtxt("./MissingDataOpenData/MissingDataOpenData/data_splits/"+split_name+".txt",dtype=str)
        self.preprocess = lambda x: x*(self.interval[1]-self.interval[0])+self.interval[0]
        self.postprocess = lambda x: (x-self.interval[0])*(1/(self.interval[1]-self.interval[0]))
        self.postprocess_with_clamp = lambda x: self.postprocess(x).clamp(0,1)
        self.interp = cv2.INTER_AREA
        self.dtype = dtype

    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self,idx):
        fn_masked = self.data_path+"masked/"+self.filenames[idx]+"_stroke_masked.png"
        fn_original = self.data_path+"originals/"+self.filenames[idx]+".jpg"
        fn_mask = self.data_path+"masks/"+self.filenames[idx]+"_stroke_mask.png"
        items = []
        for k,fn in enumerate([fn_masked,fn_original,fn_mask]):
            item = np.asarray(Image.open(fn)).astype(float)/255
            if not self.reshape_size[:2]==list(item.shape[:2]):
                item = cv2.resize(item, self.reshape_size[:2], interpolation=self.interp)
            item = torch.from_numpy(item).type(dtype=self.dtype)
            if k in [0,1]:
                item = self.preprocess(item)
            if len(item.shape)==3:
                item = item.permute(2,0,1)
            elif len(item.shape)==2:
                item = item.unsqueeze(0)
            else:
                raise ValueError("bla")
            items.append(item.to(self.device))
        image_masked,image,mask = tuple(items)
        return image_masked,image,mask

if __name__=="__main__":
    reshape_size = [64,64,3]
    train_dataset = CatDataset(split_name="training",reshape_size=reshape_size)
    vali_dataset = CatDataset(split_name="validation_1000",reshape_size=reshape_size)

    train_dataloader = torch.utils.data.DataLoader(train_dataset,batch_size=32,shuffle=True,drop_last=True)
    vali_dataloader = torch.utils.data.DataLoader(vali_dataset,batch_size=32,shuffle=True,drop_last=True)