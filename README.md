# UW-Madison-segmentation
### Brief description:
Project for UW-Madison GI Tract Image Segmentation (https://www.kaggle.com/competitions/uw-madison-gi-tract-image-segmentation)
Basic pipeline:
Train: image data loading -> rle-to-mask -> random 224x224 crops -> UNet training with dice loss
Validation: image data loading -> cropping arbitraty-shaped images into 224x224 patches -> predict mask -> avg scores for overlapping areas

### Future improvemets
- add more augmentations
- speed up data loading via caching or something similar
- try other architectures for segmentation model
- try 3D segmetation given that we have sliced data available

### How to run
Train:
`python train.py --epochs 128 -model-depth 16 -batch-size 32 --data-path ./images/for/train`
Val:
`python train.py --validate --data-path ./images/for/val -ckpt trained-model.pth --output res.csv`
