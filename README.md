# HashGAN

## Preprocess Dataset
1. Download Celeba Mask-HQ dataset [here](https://drive.google.com/file/d/1badu11NqxGf6qM3PTTooQDJvQbejgbTv/view) 


2. Unzip dataset at data/


2. Preprocess data with this code.
```
python3 data/preprocess_celeba.py data/CelebAMask-HQ
```


## Training Stage 1
```
CUDA_VISIBLE_DEVICES=0 python3 train_cnerf.py --expname EXPNAME --size 64          # Single GPU
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 train_cnerf.py --expname EXPNAME --size 64    # Multiple GPUs
```

## Training Stage 2
Make sure DO NOT use Multi GPUs!
Path Regularization does not work with Multi GPUs
```
CUDA_VISIBLE_DEVICES=0 python3 train_full.py --expname EXPAME --size 256          # Single GPU
```