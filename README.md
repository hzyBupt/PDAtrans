# Joint Parallel Modeling with Direction-Wise Convolution and Deformable Transformer for 3D Medical Image Segmentation

This is the code for paper "Joint Parallel Modeling with Direction-Wise Convolution and Deformable Transformer for 3D Medical Image Segmentation". The model is implemented with PyTorch.

# Abstruct

Three-dimensional (3D) medical images often exhibit anisotropic voxel spacing and complex organ shapes, making accurate segmentation challenging. Conventional CNNs are limited in addressing these challenges because they rely on fixed, isotropic kernels that cannot adapt to directional variations in resolution. While recent CNN-transformer hybrids have shown potential, their sequential integration of convolution and transformer blocks limits the joint modeling of local details and global context. To overcome these limitations, we propose PDAtrans (Parallel Direction-wise Aggregate Network), which constructs local and global representations in parallel through convolutional and transformer branches. The convolutional branch employs Direction-wise Aggregate Convolution (DAConv) to decouple spatial filtering across different directions, effectively addressing anisotropy. The transformer branch introduces a Direction-wise Shift Window Deformable Transformer (DST) that enables anatomy-aware adjustment of attention regions via learned offsets and enhances alignment with complex anatomical structures. Evaluated on three public 3D medical image datasets, PDAtrans achieves state-of-the-art performance with significantly lower computational cost. Ablation studies confirm the effectiveness of our direction-wise modules in enhancing segmentation under real-world constraints.

# Pipline

<div align="center">
  <img src="Figure.png" width="85%">
</div>

# Usage

## Requirements

```
git clone https://github.com/llsurreal919/ADMIT

pip install -r requirements.txt
```

## Dataset

The test dataset can be downloaded from [kaggle](https://www.kaggle.com/datasets/drxinchengzhu/kodak24) .

The training dataset can be downloaded from [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/) .

## Training

We will release the tutorial soon.

## Testing

Pre-trained models can be downloaded from [GoogleDrive](https://drive.google.com/file/d/1o7aqd5OgAIltr8NK6tmF-jkeq7xmc1HS/view?usp=sharing).

Example usage:

    python test_dyna_kodak.py

# Acknowledgement

The style of coding is borrowed from [Dynamic_JSCC](https://github.com/mingyuyng/Dynamic_JSCC) and partially built upon the [Neighborhood Attention Transformer](https://github.com/SHI-Labs/Neighborhood-Attention-Transformer). We thank the authors for sharing their codes.

# Contact

If you have any question, please contact me (He Ziyang) via heziyang@bupt.edu.cn.
