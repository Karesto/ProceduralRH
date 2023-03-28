from torch.utils.data import Dataset
import torch
import numpy as np
import os
import random
from PIL import Image
import torchvision.datasets as dset
from random import randrange
import random
import utils

Dict = {} #TODO: 
datadir = "rush.txt"


## TODO: Add random crop and random flip 
def read_img(img_url, image_size, is_train):
    img=Image.open(img_url).convert("RGB")

    x,y=img.size
    if x!=y:
        if is_train:
            matrix_length=min(x,y)
            x1=randrange(0,x-matrix_length+1)
            y1=randrange(0,y-matrix_length+1)
            img=img.crop((x1,y1,x1+matrix_length,y1+matrix_length))

    if random.random()>0.5 and is_train:
        img=img.transpose(Image.FLIP_LEFT_RIGHT)

    img=img.resize((image_size,image_size),resample=Image.BILINEAR)
    return np.array(img)


def getfilelist(path):
    all_file=[]
    for dir,folder,file in os.walk(path):
        for i in file:
            t = "%s/%s"%(dir,i)
            all_file.append(t)
    return all_file

class RushDataset(Dataset):
    #TODO : In the future, adapt for multi threadding this code. Useless for now.

    def __init__(self, pt_dataset, clusters, perm=None, mask_path=None, is_train=False, image_size=6, random_stroke=False):

        self.is_train=is_train
        self.pt_dataset = pt_dataset
        self.image_id_list = []
        

        dset_tmp=dset.ImageFolder(root=self.pt_dataset)
        url_label=dset_tmp.imgs
        for (x,y) in url_label:
            self.image_id_list.append(x)

        self.random_stroke=random_stroke
        self.clusters = clusters
        self.perm = torch.arange(image_size*image_size) if perm is None else perm

        self.mask_dir=mask_path
        self.mask_list=os.listdir(self.mask_dir)
        self.mask_list=sorted(self.mask_list)

        self.vocab_size = clusters.size(0)
        #self.block_size = 32*32 - 1
        self.block_size = image_size*image_size
        self.mask_num=len(self.mask_list)

        self.image_size=image_size

        print("# Mask is %d, # Image is %d"%(self.mask_num,len(self.image_id_list)))
        
    def __len__(self):
        return len(self.image_id_list)

    def __getitem__(self, idx):

        if self.is_train:
            selected_mask_name=random.sample(self.mask_list,1)[0]
        else:
            selected_mask_name=self.mask_list[idx%self.mask_num]
        
        if not self.random_stroke:
            selected_mask_dir=os.path.join(self.mask_dir,selected_mask_name)
            #selected_mask_dir=selected_mask_name
            mask=Image.open(selected_mask_dir).convert("L")
        else:
            mask = generate_stroke_mask([256, 256])
            mask = (mask>0).astype(np.uint8)* 255
            mask = Image.fromarray(mask).convert("L")
        
        mask = mask.resize((self.image_size,self.image_size),resample=Image.NEAREST)

        if self.is_train:
            if random.random()>0.5:
                mask=mask.transpose(Image.FLIP_LEFT_RIGHT)
            if random.random()>0.5:
                mask=mask.transpose(Image.FLIP_TOP_BOTTOM)

        mask=torch.from_numpy(np.array(mask)).view(-1)
        mask=(mask/255.)>0.5
        mask=mask.float()

        selected_img_name = self.image_id_list[idx]
        selected_img_url = selected_img_name
        #selected_img_url = os.path.join(self.pt_dataset,selected_img_name)
        x = read_img(selected_img_url,image_size=self.image_size, is_train=self.is_train)
        
        x = torch.from_numpy(np.array(x)).view(-1, 3) # flatten out all pixels
        x = x[self.perm].float() # reshuffle pixels with any fixed permutation and -> float
        a = ((x[:, None, :] - self.clusters[None, :, :])**2).sum(-1).argmin(1) # cluster assignments

        return a[:], mask[:]


def tokenizer(rush):
    return torch.tensor(utils.decoder(rush).flatten()+1)





class MyDataset(Dataset):
    def __init__(self, src, tokenizer):
        self.src = src
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.src)

    def __getitem__(self, idx):
        src = self.src[idx]
        return src

def RushDatasets(num = 5000, new = False):

    

    if os.path.exists('data/transrush.npy') and not new:
        rush = np.load('data/transrush.npy')
    else:
        
        # unsolvable = np.load("data/unsolvable_lvl1.npy")
        # unsolvable2 = np.load("data/unsolv_black_boxes_6x6translated.npy")
        # unsolvable = np.concatenate((unsolvable, unsolvable2), axis = 0)

        base = np.genfromtxt(datadir, dtype= str)[:,1]
        data = base[np.random.choice(len(base),num)]
        np.save("rushtest.npy", data)

    

    dataset = MyDataset(data, tokenizer)
    return(dataset)

    train_set, val_set = torch.utils.data.random_split(dataset, [train_length, val_length], torch.Generator().manual_seed(42))
    sampler = torch.utils.data.WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'),len(train_set))

    train_loader = torch.utils.data.DataLoader(dataset=train_set,
                                          batch_size=32,
                                          shuffle=True)
    val_loader = torch.utils.data.DataLoader(dataset=val_set,
                                          batch_size=32, 
                                          shuffle=True)
    return(train_loader, val_loader)