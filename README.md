# UG2L
This official repository is about the work "**Uncertainty-Aware Graph Reasoning with Global Collaborative Learning for Remote Sensing Salient Object Detection**" in IEEE GRSL 2023.

# Abstract
Recently, fully convolutional networks (FCNs) have contributed significantly to salient object detection in optical remote sensing images (RSIs). However, owing to the limited receptive fields of FCNs, accurate and integral detection of salient objects in RSIs with complex edges and irregular topology is still challenging. Moreover, suffering from the low contrast and complicated background of RSIs, existing models often occur ambiguous or uncertain recognition. To remedy the above problems, we propose a novel hybrid modeling approach, i.e., uncertaintyaware graph reasoning with global collaborative learning (UG2L) framework. Specifically, we propose a graph reasoning pipeline to model the intricate relations among RSI patches instead of pixels, and introduce an efficient graph reasoning block (GRB) to build graph representations. On top of it, a global context block (GCB) with a linear attention mechanism is proposed to explore the multiscale and global context collaboratively. Finally, we design a simple yet effective uncertainty-aware loss (UAL) to enhance the model’s reliability for better prediction of saliency or non-saliency. Experimental and visual results on three datasets show the superiority of the proposed UG2L.

# How to use

## Option1. Directly download the saliency maps from zip files for comparison
```
predict_smaps_UG2L_ORSSD.zip
predict_smaps_UG2L_EORSSD.zip
predict_smaps_UG2L_ORS_4199.zip
```

## Option2. Generate saliency maps via model inference

### 1. Install newest versions of torch and torchdata
```
thop                      0.0.31
tqdm                      4.59.0
numpy                     1.20.2
timm                      0.4.12
tokenizers                0.12.1
torch                     1.8.1
torchvision               0.9.1
```

### 2. Download weights files from Google Drive

<https://drive.google.com/drive/folders/1oXMfkoy6SRVd2-q3chwm5lFjugOpyA5J?usp=drive_link>


### 3. Run getsmaps.py to generate the saliency maps
```
python getsmaps.py
```
![image](https://github.com/lyf0801/UG2L/assets/73867361/630dad65-f5ca-484c-a773-31350fe4e217)

### 4. Run compute_metrics.py to calculate the qualititive results
```
python compute_results/compute_metrics.py
```
![image](https://github.com/lyf0801/UG2L/assets/73867361/d4bc521e-bc95-411b-924e-ca09db2f38f0)

## Option3. Retraining the model

**Modify the dataset and pre-training weight paths and thus run the training.py**

```
python train_UG2L.py
```

![image](https://github.com/lyf0801/UG2L/assets/73867361/23a7b4c0-d913-4804-87de-4c2c3602d64f)


# Citation (If you think this repository could help you, please cite)

```BibTeX
@ARTICLE{UG2L2023,

  author={Liu, Yanfeng and Yuan, Yuan and Wang, Qi},

  journal={IEEE Geoscience and Remote Sensening Letters},

  title={Uncertainty-Aware Graph Reasoning with Global Collaborative Learning for Remote Sensing Salient Object Detection},

  year={2023},

  volume={20},

  number={},

  pages={1-5},

}

@ARTICLE{SDNet2023,

  author={Liu, Yanfeng and Xiong, Zhitong and Yuan, Yuan and Wang, Qi},
  
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  
  title={Transcending Pixels: Boosting Saliency Detection via Scene Understanding From Aerial Imagery}, 
  
  year={2023},
  
  volume={61},
  
  number={},
  
  pages={1-16},

  doi={10.1109/TGRS.2023.3298661}

  }

@ARTICLE{SRAL2023,

  author={Liu, Yanfeng and Xiong, Zhitong and Yuan, Yuan and Wang, Qi},
  
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  
  title={Distilling Knowledge From Super-Resolution for Efficient Remote Sensing Salient Object Detection}, 
  
  year={2023},
  
  volume={61},
  
  number={},
  
  pages={1-16},
  
  doi={10.1109/TGRS.2023.3267271}
  
  }

@InProceedings{RSSOD2023,

  author = {Xiong, Zhitong and Liu, Yanfeng and Wang, Qi and Zhu, Xiao Xiang},

  title = {RSSOD-Bench: A Large-Scale Benchmark Dataset for Salient Object Detection in Optical Remote Sensing Imagery},

  booktitle = {Proc. IEEE International Geoscience and Remote Sensing Symposium (IGARSS)},

  pages={6549-6552},

  year = {2023}

}

@ARTICLE{HFANet2022,

  author={Wang, Qi and Liu, Yanfeng and Xiong, Zhitong and Yuan, Yuan},

  journal={IEEE Transactions on Geoscience and Remote Sensing},

  title={Hybrid Feature Aligned Network for Salient Object Detection in Optical Remote Sensing Imagery},

  year={2022},

  volume={60},

  number={},

  pages={1-15},

  doi={10.1109/TGRS.2022.3181062}

}
```

# Acknowledgment and our other works
1. <https://github.com/EarthNets/Dataset4EO>
2. <https://github.com/lyf0801/SDNet>
3. <https://github.com/lyf0801/HFANet>
4. <https://github.com/lyf0801/SRAL>
5. <https://github.com/rmcong/DAFNet_TIP20>
6. <https://github.com/rmcong/EORSSD-dataset>
7. <https://github.com/rmcong/ORSSD-dataset>
