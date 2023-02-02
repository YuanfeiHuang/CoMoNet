# Deep Convolution Modulation for Image Super-resolution
This repository is for CoMoNet introduced in the following paper

Yuanfei Huang, Jie Li, Yanting Hu, Hua Huang and Xinbo Gao, "Deep Convolution Modulation for Image Super-resolution", submitted.

## Overflow

![Pipeline of CoMo](/Figs/Pipeline_CoMo.png)


![Framework of CoMoNet](/Figs/Framework_CoMoNet.png)

## Dependenices
* python 3.8
* pytorch >= 1.7.0
* NVIDIA GPU + CUDA

## Data preparing
Download [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/) datasets into the path "../../Datasets/Train/DIV2K". 

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
![Visual Results](/Figs/Performance_1.png)
![Visual Results](/Figs/Performance_2.png)
