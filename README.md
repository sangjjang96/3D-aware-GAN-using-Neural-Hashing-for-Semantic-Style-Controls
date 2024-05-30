# HashGAN

![Teaser](https://github.com/sangjjang96/3D-aware-GAN-using-Neural-Hashing-for-Semantic-Style-Controls/assets/59731956/9bf57599-78f3-4361-a9ab-3b6b144f3e81)

## Preprocess Dataset
1. Download Celeba Mask-HQ dataset [here](https://drive.google.com/file/d/1badu11NqxGf6qM3PTTooQDJvQbejgbTv/view) 


2. Unzip dataset at data/


2. Preprocess data with this code.
```
python3 data/preprocess_celeba.py data/CelebAMask-HQ
```


## Training Stage 1
```
python3 train.py --expname EXPNAME --size 64                        # Single GPU
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m torch.distributed.launch --master_port 1234 --nproc_per_node 4 train.py --expname EXPNAME --size 64    # Multiple GPUs
```

## Training Stage 2
Make sure DO NOT use Multi GPUs!
Path Regularization does not work with Multi GPUs
```
python3 train.py --expname EXPNAME USED IN STAGE 1 --size 256 --pretrain_render_path ./checkpoints/EXPNAME USED IN STAGE 1/EXPNAME_vol_renderer.pt
```
