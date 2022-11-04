# MobileNet V1

PyTorch implementation of MobileNet(v1) and train script using Caltech-256 Dataset.


## Usage

### Train

1. Download [Caltech-256](https://data.caltech.edu/records/nyy15-4j048) Dataset and move to `data/caltech256` folder.
   
```
MobileNetV1/
    └ data/
        └ caltech256/
            └ 256_ObjectCategories/
                ├ 001.ak47/
                │   ├ 001_0001.jpg
                │   ⁝
                │
                ├ 002.american-flag/
                │   ├ 002_0001.jpg
                │   ⁝
                ⁝
                └ 257.clutter/
                    ├ 257_0001.jpg
                    ⁝               
```

2. Execute `tools/train.py`.

```
python tools/train.py --epoch 100 --batch_size 64
```

## Reference

- Howard, Andrew G., et al. "Mobilenets: Efficient convolutional neural networks for mobile vision applications." arXiv preprint arXiv:1704.04861 (2017).

- Griffin, Holub, & Perona. (2022). Caltech 256 (1.0) [Data set]. CaltechDATA. https://doi.org/10.22002/D1.20087