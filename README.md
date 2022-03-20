# PureT
Implementation of __End-to-End Transformer Based Model for Image Captioning__ [AAAI 2022]

![architecture](./imgs/architecture.png)

## Requirements (Our Main Enviroment)
+ Python 3.7.4
+ PyTorch 1.5.1
+ TorchVision 0.6.0
+ [coco-caption](https://github.com/tylin/coco-caption)
+ numpy
+ tqdm

## Preparation
__(some important files are uploading! slow!)__

...(waiting for description)

## Training
```
# for XE training
bash experiments_PureT/PureT_XE/train.sh

# for SCST training
bash experiments_PureT/PureT_SCST/train.sh
```

## Evaluation
__(The pretrained models are uploading! slow!)__

for example:
```
CUDA_VISIBLE_DEVICES=0 python main_test.py --folder experiments_PureT/PureT_SCST/ --resume 27
```

Loading...

___
The code is being sorted out and will be uploaded as soon as possible.
代码整理中
