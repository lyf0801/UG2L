import numpy as np
import torch
import torch.utils.data as Data
import os
from torch import nn
from torch import optim
from torch.utils.data import DataLoader

from ORSI_SOD_dataset import ORSI_SOD_Dataset
from tqdm import tqdm
from src.UG2L import net as Net 
from evaluator import Eval_thread
from PIL import Image
import time

os.environ['CUDA_VISIBLE_DEVICES'] = '2'





def unload(x):
    y = x.squeeze().cpu().data.numpy()
    return y
def convert2img(x):
    return Image.fromarray(x*255).convert('L')
def min_max_normalization(x):
    x_normed = (x - np.min(x)) / (np.max(x)-np.min(x))
    return x_normed
def save_smap(smap, path, negative_threshold=0.25):
    # smap: [1, H, W]
    if torch.max(smap) <= negative_threshold:
        smap[smap<negative_threshold] = 0
        smap = convert2img(unload(smap))
    else:
        smap = convert2img(min_max_normalization(unload(smap)))
    smap.save(path)



def getsmaps(dataset_name):
    ##define dataset
    dataset_root  = "/data/iopen/lyf/SaliencyOD_in_RSIs/" + dataset_name + " dataset/"
    test_set = ORSI_SOD_Dataset(root = dataset_root,  mode = "test", aug = False)
    test_loader = DataLoader(test_set, batch_size = 1, shuffle = True, num_workers = 1)
    
    ##define network and load weight
    net = Net().cuda().eval()  
    if dataset_name == "ORSSD":
        net.load_state_dict(torch.load("./data/weights/ORSSD_weights.pth"))  ##UG2L
    elif dataset_name == "EORSSD":
        net.load_state_dict(torch.load("./data/weights/EORSSD_weights.pth")) ##UG2L
    elif dataset_name == "ORS_4199":
        net.load_state_dict(torch.load("./data/weights/ORS_4199_weights.pth")) ##UG2L

    ##save saliency map
    infer_time = 0
    for image, label, _, name in tqdm(test_loader): 
        with torch.no_grad():
            image, label = image.cuda(), label.cuda()
            
            t1 = time.time()
            smap = net(image)  
            t2 = time.time()
            infer_time += (t2 - t1)
            
            ##if not exist then define
            dirs = "./data/output/predict_smaps" +  "_UG2L_" + dataset_name
            if not os.path.exists(dirs):
                os.makedirs(dirs)
            path = os.path.join(dirs, name[0] + "_UG2L" + '.png')  
            save_smap(smap, path)
            
    print(len(test_loader))
    print(infer_time)
    print(len(test_loader) / infer_time)  # inference speed (without I/O time),

if __name__ == "__main__":
    
    net = Net().cuda().eval()  
    
    from thop import profile
    from thop import clever_format
    x = torch.Tensor(1,3,448,448).cuda()
    macs, params = profile(net, inputs=(x, ), verbose = False)
    print('flops: ', f'{macs/1e9}GMac', 'params: ', f'{params/1e6}M')

    dataset = ["ORSSD", "EORSSD", "ORS_4199"]
    for datseti in dataset:
        getsmaps(datseti)
