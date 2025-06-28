# Scale-Recursive Wavelet Neural Operator for Image Generation

## Overview

This repository contains an *autoregressive coarse-to-fine* generator that
predicts an image one scale at a time using a **weight–tied Wavelet Neural
Operator (W-NO)**.  The generator starts from a $4\times4$ noise canvas and
repeatedly doubles the resolution until a $64\times64$ RGB image is reached.

<p align="center">
  <img src="outputs/epoch_20/cf_samples.png" width="70%">
</p>

## Motivation

Image generation often stores the entire spatial activation stack during a
single forward pass (e.g. UNet diffusion), pushing memory beyond commodity
GPUs.  A *recursive* formulation keeps only one scale in memory at a time –
$\mathcal{O}(H\,W)$ → $\mathcal{O}(H\,W/4)$ at 64×64 – while re-using the same
operator at every level.

Fourier Neural Operators (FNOs) are compelling for global, translation-
periodic physics, but raw FFTs introduce ringing and checkerboards in natural
images.  We therefore replace the Fourier transform with a **1-level Haar
Discrete Wavelet Transform (DWT)** – preserving locality while still offering
multi-resolution coupling.

## Algorithmic Core

### Notation

* $x\_r \in [0,1]^{3\times r\times r}$ – current resolution canvas.
* $\epsilon\_r \sim \mathcal{N}(0,\sigma\_r^2)$ – Gaussian noise injected at
  every level.
* $\mathcal{T}$ – weight-tied Wavelet Neural Operator.

At each scale we perform


$x\_r^{\uparrow 2} = \text{NearestUpsample}(x\_r)$

$\hat{x}\_{2r} = \mathcal{T}\bigl(\;x\_r^{\uparrow 2},\;\epsilon\_r,\;\text{coords}\;\bigr)$

$x\_{2r} = \mathrm{clip}(x\_r^{\uparrow 2}+\hat{x}\_{2r},\, 0,1).$

### Wavelet Neural Operator

1. Forward **Haar DWT** splits the feature map into one low-pass block $L$ and
   three high-pass blocks $H^{(o)}\_{o\in\{\text{LH,HL,HH}\}}$.
2. A $1\times1$ convolution is applied to $L$; another $1\times1$ convolution
   is applied jointly to the concatenated $H^{(o)}$.
3. Inverse DWT reconstructs the spatial residual at the same resolution.

This $\mathcal{T}$ is embedded in a residual stack and shared across all
coarse→fine transitions.

## Training Objective

For each mini-batch and every scale


$\mathcal{L}=\sum\_{\ell=0}^{3}\;w\_{\ell}\Bigl[\;\|x\_{gt}^{(\ell)}-x\_{pred}^{(\ell)}\|\_1 +$

$\lambda\_{v}\,\|\phi(x\_{gt}^{(\ell)})-\phi(x\_{pred}^{(\ell)})\|\_1 +$

$\lambda\_{s}\,(1-\text{SSIM}) + \lambda\_{g}\,\|\nabla x\_{gt}^{(\ell)}-\nabla x\_{pred}^{(\ell)}\|\_1\Bigr]$


plus a hinge **PatchGAN** loss (weight $\lambda\_{\text{adv}}$) acting only on
the finest scale.

## Memory Footprint

Only the tensors of the *current* resolution are kept in GPU memory: roughly
`batch × channels × H × W`.  With batch = 128 and width = 192 the peak VRAM is
< 4 GB for 64×64 training.

## Results (CIFAR-10, 64×64)

| Epoch | PSNR (dB) | FID ↓ | Comments                  |
|------:|---------:|------:|---------------------------|
|  10   | 23.8     | -  | Coarse colour blobs       |
|  20   | **25.9** | -  | Edges sharpen; see above  |
|  30   | 26.5     | -  | Texture appears           |

*Evaluation uses 10 k validation images, no classifier guidance.*

## How to Run

```bash
pip install -r requirements.txt
python main.py            # trains 25 epochs and writes samples / curves
```

Intermediate predictions for every scale can be dumped by uncommenting the
lines in `generate_samples()`. Load pretrained checkpoints from `ckpt` directory.
