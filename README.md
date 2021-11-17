# Image-specific Convolutional Kernel Modulation for Single Image Super-resolution
This repository is for IKM introduced in the following paper

Yuanfei Huang, Jie Li, Yanting Hu, Hua Huang and Xinbo Gao*, "Image-specific Convolutional Kernel Modulation for Single Image Super-resolution", arXiv preprint arXiv:2111.08362(2021)

[arXiv](https://arxiv.org/abs/2111.08362)

## Overflow

![Pipeline of IKM](/Figs/Pipeline_IKM.png)


![Framework of UHDN](/Figs/Framework_UHDN.png)

## Dependenices
* python 3.8
* pytorch >= 1.7.0
* NVIDIA GPU + CUDA

## Data preparing
Download [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/) and [Flickr2K](http://cv.snu.ac.kr/research/EDSR/Flickr2K.tar) datasets into the path "../../Datasets/Train/DF2K". 

## Train
1. Replace the train dataset path '../../Datasets/Train/' and validation dataset '../../Datasets/Test/' with your training and validation datasets, respectively.

3. Set the configurations in 'option.py' as you want.

 ```bash
python main.py --train 'Train'
```

## Test
1. Download models from [Google Drive](https://drive.google.com/drive/folders/10zIyGhTENJtfJytHoZHwf7QRrUsErl_C?usp=sharing) or [BaiduYun](https://pan.baidu.com/s/1cx_wCIXWYMQO02-78k9Gcw)(password: 06v3).

2. Replace the test dataset path '../../Datasets/Test/' with your datasets.

```bash
python main.py --train 'Test'
```

## Results
### Quantitative Results (PSNR/SSIM)
![Quantitative Results](/Figs/Performance_Table.png)

### Qualitative Results
![Fig.6](/Figs/Performance_Fig.png)

## Citation
```
@misc{huang2021imagespecific,
      title={Image-specific Convolutional Kernel Modulation for Single Image Super-resolution}, 
      author={Yuanfei Huang and Jie Li and Yanting Hu and Xinbo Gao and Hua Huang},
      year={2021},
      eprint={2111.08362},
      archivePrefix={arXiv},
      primaryClass={eess.IV}
}
```
