# ImageGenerationDiffusionModels

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://paul-vdl.github.io/ImageGenerationDiffusionModels.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://paul-vdl.github.io/ImageGenerationDiffusionModels.jl/dev/)
[![Build Status](https://github.com/paul-vdl/ImageGenerationDiffusionModels.jl/actions/workflows/CI.yml/badge.svg?branch=master)](https://github.com/paul-vdl/ImageGenerationDiffusionModels.jl/actions/workflows/CI.yml?query=branch%3Amaster)
[![Coverage](https://codecov.io/gh/paul-vdl/ImageGenerationDiffusionModels.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/paul-vdl/ImageGenerationDiffusionModels.jl)

The objective of this package is to generate images using a diffusion model.

We begin by progressively adding Gaussian noise to an input image. Then, we implement a U-Net architecture and train it to predict the noise that was added. This enables the model to learn how to reverse the diffusion process and reconstruct the original image.

Ultimately, the model becomes capable of generating a clean image from pure random noise by denoising it step by step—closely following the typical pipeline used in modern diffusion-based generative models.

## Available Functions

- `ImageGenerationDiffusionModels.generate_grid()`  
  Imports the image `SyntheticImages500.mat` and converts it to a PNG file (`grid.png`). Also returns an array used by other functions.

- `ImageGenerationDiffusionModels.apply_noise(img; num_noise_steps = 500, beta_min = 0.0001, beta_max = 0.02)`  
  Applies Gaussian noise to an image array gradually. Produces `noisy_img.png` and returns the noisy array.

- `ImageGenerationDiffusionModels.train_brain(num_steps::Int = 100)`  
  Trains the model to map noisy → clean images.

- `ImageGenerationDiffusionModels.denoise_image(noisy_img::AbstractMatrix{<:Real})`  
  De-noises a noisy image using the trained neural network.

- `ImageGenerationDiffusionModels.generate_image_from_noise()`  
  Generates a new image from random noise and de-noises it.

---

## Getting Started

Here's a quick example to get started:

In Pkg-mode:
```
add https://github.com/paul-vdl/ImageGenerationDiffusionModels.jl
```
Then in REPL:
```julia
using ImageGenerationDiffusionModels
```
```julia
img = generate_grid()
```
```julia
noisy_img = apply_noise(img)
```
```julia
train_brain()
```
```julia
denoised_img = denoise_image(img[1:32, 1:32])
```
```julia
new_img = generate_image_from_noise()
```
! The two last functions create a png with the same name so you can only see one at the time
