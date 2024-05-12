from pickle import FALSE
import numpy as np
import os
from PIL import Image
from evaluator import *
from dataset import ORSSD, EORSSD, ORS_4199
from torch.utils.data import DataLoader





if __name__ == '__main__':  
    method_names = ['UG2L']
    print("一共" + str(len(method_names)) + "种对比算法")
    dataset_name = ['ORSSD', "EORSSD", "ORS_4199"]
    for method_name in method_names:
        for dataseti in dataset_name:
            root = r"D:/optical_RSIs_SOD/" + dataseti + " dataset"
            smap_path = "./data/output/predict_smaps_" + method_name + "_" + dataseti + "/"
            prefixes = [line.strip() for line in open(os.path.join(root, 'test.txt'))]
            image_paths = [os.path.join(root, 'images', prefix + '.jpg') for prefix in prefixes]

            if dataseti == "ORSSD":
                test_set = ORSSD(root = root, mode = "test", aug = False)
            elif dataseti == "EORSSD":
                test_set = EORSSD(root = root, mode = "test", aug = False)
            elif dataseti == "ORS_4199":
                test_set = ORS_4199(root = root, mode = "test", aug = False)

            test_loader = DataLoader(test_set, batch_size = 1, shuffle = True, num_workers = 1)  

            thread = Eval_thread(smap_path=smap_path, loader = test_loader, method = method_name, dataset = dataseti, output_dir = "./data/", cuda=False)
            logg, fm = thread.run()
            print(logg)


