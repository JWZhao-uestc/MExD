# MExD
This is an official repository for MExD

# Data preprocessing
All Data was preprocessed, followed by [CLAM](https://github.com/mahmoodlab/CLAM "CLAM"). Please refer to the website for more details. 

**1. Extract patches**
```javascript
python create_patches_fp.py --source DATA_DIRECTORY --save_dir RESULTS_DIRECTORY --patch_size 256 --seg --process_list CSV_FILE_NAME --patch --stitch
```
**2.Extract features** 
```javascript
python extract_features_fp.py --data_h5_dir DIR_TO_COORDS --data_slide_dir DATA_DIRECTORY --csv_path CSV_FILE_NAME --feat_dir FEATURES_DIRECTORY --batch_size 512 --slide_ext .svs
```
We use two types of feature extractors: CtransPath and ViT, please download the pretrained weights [TransPath](https://github.com/Xiyue-Wang/TransPath "TransPath") and use the provided modified "timm" package.
# Training
# training moe:
Training the MOE model, we encourage you to train the MOE based on a well-trained origin transMIL.
```javascript
bash train_moe.sh
```

# Acknolwledge
We thank the contribution of TransMIL, IBMIL, [CLAM](https://github.com/mahmoodlab/CLAM "CLAM") to the WSI-C community.

# Reference
```javascript
@InProceedings{Zhao_2025_CVPR,
    author    = {Zhao, Jianwei and Li, Xin and Yang, Fan and Zhai, Qiang and Luo, Ao and Zhao, Yang and Cheng, Hong and Fu, Huazhu},
    title     = {MExD: An Expert-Infused Diffusion Model for Whole-Slide Image Classification},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2025},
    pages     = {20789-20799}
}
```

